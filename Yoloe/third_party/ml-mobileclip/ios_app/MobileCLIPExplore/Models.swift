//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import CoreML
import Foundation

protocol CLIPEncoder {

    var targetImageSize: CGSize { get }

    func load() async

    func encode(image: CVPixelBuffer) async throws -> MLMultiArray

    func encode(text: MLMultiArray) async throws -> MLMultiArray

}

public struct ModelConfiguration: Identifiable, Hashable {
    public let name: String
    let factory: () -> CLIPEncoder
    public var id: String { name }

    public static func == (lhs: ModelConfiguration, rhs: ModelConfiguration) -> Bool {
        lhs.name == rhs.name
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(name)
    }
}

public let models: [ModelConfiguration] = [
    .init(name: "MobileCLIP-S0", factory: { S0Model() }),
    .init(name: "MobileCLIP-S1", factory: { S1Model() }),
    .init(name: "MobileCLIP-S2", factory: { S2Model() }),
    .init(name: "MobileCLIP-BLT", factory: { BLTModel() }),
]

public let defaultModel = ModelConfiguration(name: "MobileCLIP-S2", factory: { S2Model() })

public struct S0Model: CLIPEncoder {

    let imageEncoder = AsyncFactory {
        do {
            return try mobileclip_s0_image()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    let textEncoder = AsyncFactory {
        do {
            return try mobileclip_s0_text()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    func load() async {
        async let t = textEncoder.get()
        async let i = imageEncoder.get()
        _ = await (t, i)
    }

    let targetImageSize = CGSize(width: 256, height: 256)

    func encode(image: CVPixelBuffer) async throws -> MLMultiArray {
        try await imageEncoder.get().prediction(image: image).final_emb_1
    }

    func encode(text: MLMultiArray) async throws -> MLMultiArray {
        try await textEncoder.get().prediction(text: text).final_emb_1
    }
}

public struct S1Model: CLIPEncoder {

    let imageEncoder = AsyncFactory {
        do {
            return try mobileclip_s1_image()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    let textEncoder = AsyncFactory {
        do {
            return try mobileclip_s1_text()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    func load() async {
        async let t = textEncoder.get()
        async let i = imageEncoder.get()
        _ = await (t, i)
    }

    let targetImageSize = CGSize(width: 256, height: 256)

    func encode(image: CVPixelBuffer) async throws -> MLMultiArray {
        try await imageEncoder.get().prediction(image: image).final_emb_1
    }

    func encode(text: MLMultiArray) async throws -> MLMultiArray {
        try await textEncoder.get().prediction(text: text).final_emb_1
    }
}

public struct S2Model: CLIPEncoder {

    let imageEncoder = AsyncFactory {
        do {
            return try mobileclip_s2_image()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    let textEncoder = AsyncFactory {
        do {
            return try mobileclip_s2_text()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    func load() async {
        async let t = textEncoder.get()
        async let i = imageEncoder.get()
        _ = await (t, i)
    }

    let targetImageSize = CGSize(width: 256, height: 256)

    func encode(image: CVPixelBuffer) async throws -> MLMultiArray {
        try await imageEncoder.get().prediction(image: image).final_emb_1
    }

    func encode(text: MLMultiArray) async throws -> MLMultiArray {
        try await textEncoder.get().prediction(text: text).final_emb_1
    }
}

public struct BLTModel: CLIPEncoder {

    let imageEncoder = AsyncFactory {
        do {
            return try mobileclip_blt_image()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    let textEncoder = AsyncFactory {
        do {
            return try mobileclip_blt_text()
        } catch {
            fatalError("Failed to initialize ML model: \(error)")
        }
    }

    func load() async {
        async let t = textEncoder.get()
        async let i = imageEncoder.get()
        _ = await (t, i)
    }

    let targetImageSize = CGSize(width: 224, height: 224)

    func encode(image: CVPixelBuffer) async throws -> MLMultiArray {
        try await imageEncoder.get().prediction(image: image).final_emb_1
    }

    func encode(text: MLMultiArray) async throws -> MLMultiArray {
        try await textEncoder.get().prediction(text: text).final_emb_1
    }
}
