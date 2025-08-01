//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import AVFoundation
import CoreML
import SwiftUI

/// controls window for the performance measurements -- adjust up if the
/// fps numbers are too noisy
private let inferenceTimeExponentialMovingAverageWindow: Double = 10

/// run the inference at a lower frame rate, e.g. 2 == every other frame.  this reduces
/// power consumption if needed
private let inferenceFrameReduction = 0

// MARK: - Landing View
/// Landing page for selecting a prompt and viewing results
struct InferenceView: View {

    let camera: CameraController

    /// binding to flip to front/back camera
    @Binding var backCamera: Bool

    /// Hold reference to zero-shot predictor
    @State var modelConfiguration = defaultModel
    @State var zsclassifier = ZSImageClassification(model: defaultModel.factory())

    /// Prompt and classnames related
    @State private var prompt = presets[0].prompt
    @State private var textEmbeddings: [MLMultiArray] = []

    @State private var displayPredictions = [DisplayPrediction]()

    /// Inference timing, see also inferenceTimeExponentialMovingAverageWindow
    @State private var inferenceTime: CFTimeInterval = 0

    /// stream of frames -> VideoFrameView, see distributeVideoFrames
    @State private var framesToDisplay: AsyncStream<CVImageBuffer>?

    /// Controls the info popover
    @State private var showingInfo = false

    /// Shows spinner / prevents double submit
    @State private var loading = false

    var body: some View {
        NavigationStack {
            VStack {
                Group {
                    // top line.  lay these out on left / center / right without
                    // concern for the size of each item.  zstack lets us place them
                    // in the same place and the spacers position them as desired
                    ZStack(alignment: .leading) {

                        // display FPS and response time
                        PerformanceView(averageFrameTime: inferenceTime)

                        // title and model selection
                        HStack {
                            Spacer()
                            VStack(spacing: 0) {
                                Text("MobileCLIP")
                                    .font(.title)
                                    .fontWeight(.heavy)

                                Menu {
                                    Picker("", selection: $modelConfiguration) {
                                        ForEach(models, id: \.self) { model in
                                            Text(model.name)
                                        }
                                    }
                                } label: {
                                    HStack {
                                        Text("\(modelConfiguration.name)")
                                        Image(systemName: "chevron.down")
                                    }
                                    .font(.caption)
                                }
                            }
                            Spacer()
                        }

                        // information panel
                        HStack {
                            Spacer()

                            Button(action: { showingInfo.toggle() }) {
                                Image(systemName: "info.circle")
                            }
                            .popover(isPresented: $showingInfo) {
                                ScrollView(.vertical) {
                                    InfoView()
                                        .overlay(alignment: .topTrailing) {
                                            Button(action: { showingInfo.toggle() }) {
                                                Image(systemName: "x.circle")
                                                    .padding()
                                                    .foregroundColor(.blue)
                                            }
                                            .buttonStyle(.plain)
                                        }
                                }
                            }
                            .buttonStyle(.plain)
                            .padding(.trailing, 8)
                        }
                    }
                    .padding(.vertical, 5)

                    // Camera viewport display
                    VStack {
                        if let framesToDisplay {
                            VideoFrameView(
                                frames: framesToDisplay, backCamera: $backCamera)
                        }
                    }
                    .frame(height: 412)

                    // task to distribute video frames -- this will cancel
                    // and restart when the view is on/off screen.  note: it is
                    // important that this is here (attached to the VideoFrameView)
                    // rather than the outer view because this has the correct lifecycle
                    .task {
                        if Task.isCancelled {
                            return
                        }

                        await distributeVideoFrames()
                    }

                    // Select preset prompt
                    HStack {
                        ForEach(presets) { preset in
                            Button(action: { setPrompt(preset.prompt) }) {
                                Text(preset.title)
                            }
                            .buttonStyle(PresetButton())
                            .disabled(loading)
                        }
                    }
                    .padding(.vertical, 5)

                    // Display and edit prompt
                    Group {
                        HStack {
                            Text("PROMPT")
                                .font(.callout)
                                .fontWeight(.semibold)
                                .foregroundStyle(.secondary)

                            Spacer()
                        }
                        .padding(.top, 15.0)

                        NavigationLink {
                            PromptEditor(prompt: $prompt)
                        } label: {
                            HStack {
                                PromptPreview(prompt: prompt)

                                Spacer()

                                Image(systemName: "pencil")
                                    .padding(.leading, 8)
                            }
                        }
                    }
                    .padding(.horizontal, 15.0)
                    .padding(.bottom, 6)
                    .task {
                        // when coming on screen (e.g. back from the editor)
                        // update the embeddings
                        self.rebuildEmbeddings()
                    }
                    .onChange(of: modelConfiguration) { oldValue, newValue in
                        Task {
                            await self.setModel(newValue)
                        }
                    }
                }
                .padding(.horizontal, 8)

                // prediction results
                if loading {
                    Spacer()
                    ProgressView()
                        .controlSize(.large)
                    Spacer()
                } else {
                    PredictionsTable(displayPredictions: displayPredictions)
                        .padding(.top, 10.0)
                }
            }
        }
        .font(.body)
        .task {
            // trigger the model loads (async)
            await zsclassifier.load()
        }
    }

    /// consume video frames from the camera and distribute to the live view and the classifier
    func distributeVideoFrames() async {
        // attach a stream to the camera -- this code will read this
        let frames = AsyncStream<CMSampleBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            camera.attach(continuation: $0)
        }

        // frames -> classifier
        var framesToClassifyContinuation: AsyncStream<CVImageBuffer>.Continuation!
        let framesToClassify = AsyncStream<CVImageBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            framesToClassifyContinuation = $0
        }

        // frames -> VideoFrameView
        var framesToDisplayContinuation: AsyncStream<CVImageBuffer>.Continuation!
        let framesToDisplay = AsyncStream<CVImageBuffer>(bufferingPolicy: .bufferingNewest(1)) {
            framesToDisplayContinuation = $0
        }
        self.framesToDisplay = framesToDisplay

        guard let framesToClassifyContinuation, let framesToDisplayContinuation else {
            print("failed to attach continuations")
            return
        }

        // set up structured tasks (important -- this means the child tasks
        // are cancelled when the parent is cancelled)
        async let distributeFrames: () = {
            [framesToClassifyContinuation, framesToDisplayContinuation] in
            for await sampleBuffer in frames {
                if let frame = sampleBuffer.imageBuffer {
                    framesToClassifyContinuation.yield(frame)
                    framesToDisplayContinuation.yield(frame)
                }
            }

            // detach from the camera controller and feed to the video view
            await MainActor.run {
                self.framesToDisplay = nil
                self.camera.detatch()
            }

            framesToClassifyContinuation.finish()
            framesToDisplayContinuation.finish()
        }()

        // classification runs async from live frames
        async let classifyFrames: () = {
            var count = 0
            for await frame in framesToClassify {
                if count == inferenceFrameReduction {
                    await performZeroShotClassification(frame)
                    count = 0
                } else {
                    count += 1
                }
            }
        }()

        await distributeFrames
        await classifyFrames
    }

    func setModel(_ model: ModelConfiguration) async {
        await self.zsclassifier.setModel(model)
        rebuildEmbeddings()
    }

    func rebuildEmbeddings() {
        self.setPrompt(self.prompt)
    }

    /// Construct prompt and text embeddings
    func setPrompt(_ prompt: Prompt) {
        if loading {
            // if the model isn't loaded yet someone might try and submit
            // multiple times -- it can take a few seconds to load
            return
        }

        self.loading = true
        Task {
            // wait for both the text and image model to be ready, otherwise
            // we may remove the spinner before we can actually start inference
            await zsclassifier.load()
            textEmbeddings = await zsclassifier.computeTextEmbeddings(
                promptArr: prompt.fullPrompts())
            self.prompt = prompt
            self.loading = false
        }
    }

    /// Compute image embeddings and cosine similarities with text embedding
    func performZeroShotClassification(_ frame: CVPixelBuffer) async {
        if prompt.classNames.isEmpty {
            displayPredictions = []
        } else {
            guard let output = await zsclassifier.computeImageEmbeddings(frame: frame) else {
                return
            }

            let imageEmbedding = output.embedding
            observeTiming(output.interval)

            self.displayPredictions = zip(textEmbeddings, prompt.classNames)
                .map { (textEmbedding, className) in
                    let similarity = zsclassifier.cosineSimilarity(imageEmbedding, textEmbedding)
                    return DisplayPrediction(className: className, cosineSimilarity: similarity)
                }
        }
    }

    func observeTiming(_ value: CFTimeInterval) {
        // discard values larger than 500 milliseconds that sometimes appear during start up
        guard (value * 1000) < 500 else { return }

        if inferenceTime == 0 {
            inferenceTime = value
        } else {
            // an exponential decay moving average
            inferenceTime =
                inferenceTime
                - (inferenceTime / inferenceTimeExponentialMovingAverageWindow)
                + (value / inferenceTimeExponentialMovingAverageWindow)
        }
    }
}

// MARK: - Helpers
/// Displays the current FPS and response time classnames are being detected at
private struct PerformanceView: View {

    let averageFrameTime: CFTimeInterval
    let numberWidth: CGFloat = 30
    let labelWidth: CGFloat = 30

    var body: some View {
        VStack(alignment: .trailing) {
            if averageFrameTime > 0 {
                let mt = ((averageFrameTime * 1000 * 10).rounded() / 10)
                let fps = (1 / averageFrameTime).rounded()

                HStack {
                    // fixed sizes so things don't move around as the numbers change
                    Text("\(fps.formatted())")
                        .frame(width: numberWidth)
                    Text("FPS")
                        .frame(width: labelWidth, alignment: .leading)
                        .offset(x: -8)
                }
                HStack {
                    Text("\(mt.formatted())")
                        .frame(width: numberWidth)
                    Text("ms")
                        .frame(width: labelWidth, alignment: .leading)
                        .offset(x: -8)
                }
            }
        }
        .fontWeight(.light)
        .font(.system(size: 14))
    }

}

/// A reusable button
struct PresetButton: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.callout)
            .padding(EdgeInsets(top: 3, leading: 20, bottom: 3, trailing: 20))
            .background(isEnabled ? Color("Preset") : Color("PresetDisabled"))
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

// MARK: - Previews
#Preview {
    VStack {
        HStack {
            Button(action: {}) {
                Text("Button 1")
            }
            .buttonStyle(PresetButton())

            Button(action: {}) {
                Text("Button 2")
            }
            .buttonStyle(PresetButton())
            .disabled(true)
        }

        PerformanceView(averageFrameTime: 0.0065)

        HStack {
            ForEach(presets) { preset in
                Button(action: {}) {
                    Text(preset.title)
                }
                .buttonStyle(PresetButton())
            }
        }
    }
}
