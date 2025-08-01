//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import CoreML
import UIKit

/// shared tokenizer for all model types
private let tokenizer = AsyncFactory {
    CLIPTokenizer()
}

actor ZSImageClassification: ObservableObject {

    private let ciContext = CIContext()
    private var model: any CLIPEncoder

    public init(model: any CLIPEncoder) {
        self.model = model
    }

    func load() async {
        async let t = tokenizer.get()
        async let m = model.load()
        _ = await (t, m)
    }

    public func setModel(_ model: ModelConfiguration) {
        self.model = model.factory()
    }

    // Compute Text Embeddings
    func computeTextEmbeddings(promptArr: [String]) async -> [MLMultiArray] {
        var textEmbeddings: [MLMultiArray] = []
        do {
            for singlePrompt in promptArr {
                print("")
                print("Prompt text: \(singlePrompt)")

                // Tokenize the text query
                let inputIds = await tokenizer.get().encode_full(text: singlePrompt)

                // Convert [Int] to MultiArray
                let inputArray = try MLMultiArray(shape: [1, 77], dataType: .int32)
                for (index, element) in inputIds.enumerated() {
                    inputArray[index] = NSNumber(value: element)
                }

                // Run the text model on the text query
                let output = try await model.encode(text: inputArray)
                textEmbeddings.append(output)
            }
        } catch {
            print(error.localizedDescription)
        }
        return textEmbeddings
    }

    // Compute Image Embeddings
    func computeImageEmbeddings(frame: CVPixelBuffer) async -> (
        embedding: MLMultiArray, interval: CFTimeInterval
    )? {
        // prepare the image
        var image: CIImage? = CIImage(cvPixelBuffer: frame)
        image = image?.cropToSquare()
        image = image?.resize(size: model.targetImageSize)

        guard let image else {
            return nil
        }

        // output buffer
        let extent = image.extent
        let pixelFormat = kCVPixelFormatType_32ARGB
        var output: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(extent.width), Int(extent.height), pixelFormat, nil, &output)

        guard let output else {
            print("failed to create output CVPixelBuffer")
            return nil
        }

        ciContext.render(image, to: output)

        // Run image embedding
        do {
            let startTimer = CACurrentMediaTime()
            let output = try await model.encode(image: output)
            let endTimer = CACurrentMediaTime()
            let interval = endTimer - startTimer
            return (embedding: output, interval: interval)
        } catch {
            print(error.localizedDescription)
            return nil
        }
    }

    // Compute cosine similarity between embeddings
    nonisolated func cosineSimilarity(_ embedding1: MLMultiArray, _ embedding2: MLMultiArray)
        -> Float
    {

        // read the values out of the MLMultiArray in bulk
        let e1 = embedding1.withUnsafeBufferPointer(ofType: Float.self) { ptr in
            Array(ptr)
        }
        let e2 = embedding2.withUnsafeBufferPointer(ofType: Float.self) { ptr in
            Array(ptr)
        }

        // Get the dot product of the two embeddings
        let dotProduct: Float = zip(e1, e2).reduce(0.0) { $0 + $1.0 * $1.1 }

        // Get the magnitudes of the two embeddings
        let magnitude1: Float = sqrt(e1.reduce(0) { $0 + pow($1, 2) })
        let magnitude2: Float = sqrt(e2.reduce(0) { $0 + pow($1, 2) })

        // Get the cosine similarity
        let similarity = dotProduct / (magnitude1 * magnitude2)
        return similarity
    }
}
