//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

// MARK: - Predictions Results Table
/// Table showing predictions in sorted order
struct PredictionsTable: View {

    let displayPredictions: [DisplayPrediction]

    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    private var isCompact: Bool { horizontalSizeClass == .compact }

    /// sort keys for classes
    @State private var sortCosineSimilarity = true
    @State private var sortOrder = [
        KeyPathComparator(\DisplayPrediction.cosineSimilarity, order: .reverse)
    ]

    @Environment(\.colorScheme) var colorScheme

    var body: some View {
        VStack(spacing: 1) {

            let predictions = displayPredictions.sorted(using: sortOrder)
            let formatter = DisplayPredictionFormatter(
                predictions: predictions, colorScheme: colorScheme)

            HStack {
                Text("CLASSNAME")
                Spacer()
                Text("COSINE SIMILARITY")
            }
            .font(.footnote)
            .fontWeight(.semibold)
            .padding(.horizontal, 23.0)
            .padding(.bottom, 10)
            .foregroundStyle(.secondary)

            Divider().overlay(.gray)

            List(predictions) { prediction in

                // a prediction in a row
                HStack {
                    Text(prediction.className)
                    Spacer()
                    Text(prediction.cosineSimilarity.formatted())
                }
                .bold(formatter.isBold(prediction))
                .foregroundStyle(formatter.color(prediction))
            }
            .listStyle(PlainListStyle())
        }
    }
}
