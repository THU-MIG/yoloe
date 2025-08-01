//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

/// Displays a stream of video frames
struct VideoFrameView: View {

    let frames: AsyncStream<CVImageBuffer>

    @Binding var backCamera: Bool

    @State private var videoFrame: CVImageBuffer?

    var body: some View {
        Group {
            if let videoFrame {
                // display the image, cropped to a square, with rounded corners
                _ImageView(image: videoFrame)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .allowsHitTesting(false)
                    .padding(.horizontal, 2)

                    // control to flip to front/back facing camera
                    .overlay(alignment: .bottomTrailing) {
                        Button(action: toggleCamera) {
                            Image(systemName: "arrow.triangle.2.circlepath.circle")
                                .foregroundStyle(.blue)
                                .frame(width: 20, height: 20)
                                .padding(20)
                        }
                        .buttonStyle(.plain)
                    }

            } else {
                // spinner before the camera comes up
                ProgressView()
                    .controlSize(.large)
            }
        }
        .task {
            // feed frames to the _ImageView
            if Task.isCancelled {
                return
            }
            for await frame in frames {
                self.videoFrame = frame
            }
        }
    }

    func toggleCamera() {
        backCamera.toggle()
    }
}

/// Internal view to display a CVImageBuffer
private struct _ImageView: UIViewRepresentable {

    let image: Any
    var gravity = CALayerContentsGravity.center

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.layer.contentsGravity = gravity
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        uiView.layer.contents = image
    }
}
