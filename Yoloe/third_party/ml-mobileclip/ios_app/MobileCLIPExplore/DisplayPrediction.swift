//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

struct DisplayPrediction: Identifiable {
    var id: String { className }

    var className: String
    var cosineSimilarity: Float
}

/// Top N percent will be bold
private let boldPercent: Float = 0.05

/// It has to exceed this to be bold
private let boldThreshold: Float = 0.22

/// Colors for interpolating based on color scheme
private func colors(for colorScheme: ColorScheme) -> [Color] {
    var result = [Color]()

    let topGray: CGFloat = colorScheme == .dark ? 1.0 : 0.0
    let bottomGray: CGFloat = colorScheme == .dark ? 0.2 : 0.8

    for position in stride(from: 0, through: 1, by: 0.05) {
        let gray = (bottomGray + CGFloat(position) * (topGray - bottomGray))

        result.append(.init(cgColor: CGColor(gray: gray, alpha: 1.0)))
    }

    return result
}

/// Boldness threshold
private let maximumThreshold: Float = 0.22
private let minimumTreshold: Float = 0.15

struct DisplayPredictionFormatter {

    let actualMaximumCosineSimilarity: Float
    let colorScheme: ColorScheme

    init(predictions: [DisplayPrediction], colorScheme: ColorScheme) {
        self.actualMaximumCosineSimilarity = predictions.map { $0.cosineSimilarity }.max() ?? 0
        self.colorScheme = colorScheme
    }

    func isBold(_ prediction: DisplayPrediction) -> Bool {
        prediction.cosineSimilarity >= boldThreshold
            && prediction.cosineSimilarity >= actualMaximumCosineSimilarity * (1 - boldPercent)
    }

    func color(_ prediction: DisplayPrediction) -> Color {
        let position =
            min(max(prediction.cosineSimilarity - minimumTreshold, 0), maximumThreshold)
            / (maximumThreshold - minimumTreshold)
        let index = min(
            Int(round(Float(colors(for: colorScheme).count - 1) * position)),
            colors(for: colorScheme).count - 1)
        return colors(for: colorScheme)[index]
    }
}
