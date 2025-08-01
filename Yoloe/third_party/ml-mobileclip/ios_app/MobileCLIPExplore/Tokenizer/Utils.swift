//
//  Utils.swift
//  AudioBoloss
//
//  Created by Julien Chaumond on 07/01/2019.
//  Copyright © 2019 Hugging Face. All rights reserved.
//
// See https://github.com/huggingface/swift-coreml-transformers

import Foundation

struct Utils {
    /// Time a block in ms
    static func time<T>(label: String, _ block: () -> T) -> T {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = block()
        let diff = (CFAbsoluteTimeGetCurrent() - startTime) * 1_000
        print("[\(label)] \(diff)ms")
        return result
    }

    /// Time a block in seconds and return (output, time)
    static func time<T>(_ block: () -> T) -> (T, Double) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let result = block()
        let diff = CFAbsoluteTimeGetCurrent() - startTime
        return (result, diff)
    }

    /// Return unix timestamp in ms
    static func dateNow() -> Int64 {
        // Use `Int` when we don't support 32-bits devices/OSes anymore.
        // Int crashes on iPhone 5c.
        return Int64(Date().timeIntervalSince1970 * 1000)
    }

    /// Clamp a val to [min, max]
    static func clamp<T: Comparable>(_ val: T, _ vmin: T, _ vmax: T) -> T {
        return min(max(vmin, val), vmax)
    }

    /// Fake func that can throw.
    static func fakeThrowable<T>(_ input: T) throws -> T {
        return input
    }

    /// Substring
    static func substr(_ s: String, _ r: Range<Int>) -> String? {
        let stringCount = s.count
        if stringCount < r.upperBound || stringCount < r.lowerBound {
            return nil
        }
        let startIndex = s.index(s.startIndex, offsetBy: r.lowerBound)
        let endIndex = s.index(startIndex, offsetBy: r.upperBound - r.lowerBound)
        return String(s[startIndex ..< endIndex])
    }

    /// Invert a (k, v) dictionary
    static func invert<K, V>(_ dict: [K: V]) -> [V: K] {
        var inverted: [V: K] = [:]
        for (k, v) in dict {
            inverted[v] = k
        }
        return inverted
    }
}
