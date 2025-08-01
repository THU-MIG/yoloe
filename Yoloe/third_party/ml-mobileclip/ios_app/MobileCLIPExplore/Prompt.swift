//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation

public let presets = [
    PromptPreset(
        title: "Desk items",
        prompt: .init(
            prefix: "A photo of",
            suffix: "",
            classNames: [
                "pen",
                "pencil",
                "paper",
                "mouse",
                "keyboard",
                "computer",
                "phone",
                "stapler",
                "cup",
            ])
    ),
    PromptPreset(
        title: "Expressions",
        prompt: .init(
            prefix: "A person",
            suffix: "",
            classNames: [
                "smiling",
                "waving",
                "giving a thumbs up",
                "sticking out their tongue",
                "looking angry",
            ])
    ),
    PromptPreset(
        title: "Custom",
        prompt: .init(
            prefix: "A photo of",
            suffix: "",
            classNames: [])
    ),
]

public struct PromptPreset: Identifiable {
    public let id = UUID()
    public let title: String
    public let prompt: Prompt
}

public struct Prompt {
    public var prefix: String
    public var suffix: String
    public var classNames: [String]

    public func fullPrompts() -> [String] {
        classNames.map {
            "\(prefix) \($0) \(suffix)"
        }
    }
}
