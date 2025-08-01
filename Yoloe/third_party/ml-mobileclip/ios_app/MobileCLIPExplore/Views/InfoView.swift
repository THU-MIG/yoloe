//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

/// About MobileCLIP, model options and credits
struct InfoView: View {

    var modelOptionsData = [
        (name: "MobileCLIP-S0", latency: "2.1", accuracy: "58.1"),
        (name: "MobileCLIP-S1", latency: "5.8", accuracy: "61.3"),
        (name: "MobileCLIP-S2", latency: "6.9", accuracy: "63.7"),
        (name: "MobileCLIP-B (LT)", latency: "13.7", accuracy: "65.8"),
    ]

    var body: some View {
        VStack(spacing: 10) {
            Text("Information")
                .font(.headline)
                .bold()
                .padding(.top)

            // About MobileCLIP
            Group {
                Text("MobileCLIP")
                    .font(.largeTitle)
                    .bold()
                    .padding(.top)

                Group {
                    Text("MobileCLIP¹")
                        .fontWeight(.bold)
                    + Text(
                        " is a new family of efficient image-text models optimized for runtime performance, trained with a novel and efficient training approach, namely multi-model reinforced training."
                    )

                    Text(
                        "This app demonstrates the use of **MobileCLIP** models for performing real-time zero-shot scene classification. Users are free to customize the prompt and provide classnames of their choice."
                    )
                }
                .padding(.bottom)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)
            
            
            // Model Options Table
            Group {
                Text("Model Options")
                    .font(.title2)
                    .bold()

                Text("You can select to run any of the following MobileCLIP model options:")
                    .padding(.bottom)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal)

            Group {
                HStack {
                    Text("Name")
                    Spacer()
                        .frame(width: 105, alignment: .leading)
                    Text("Ideal Latency")
                    Spacer()
                    Text("Accuracy")
                }
                .font(.headline)
                .fontWeight(.bold)
                .padding(.horizontal)

                Divider()

                ForEach(modelOptionsData, id: \.name) { option in
                    HStack {
                        Text(option.name)
                            .fontWeight(.semibold)
                            .frame(width: 125, alignment: .leading)
                        Spacer()
                        Text("\(option.latency) ms")
                            .fontWeight(.light)
                        Spacer()
                        Text("\(option.accuracy)%")
                            .fontWeight(.light)
                    }
                    .padding(.vertical, 5)
                    .padding(.horizontal)
                    Divider()
                }
            }

            // Authors and Citations
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Spacer().frame(width: 12)
                    Text(
                        "**¹ MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training.** (CVPR 2024)"
                    )
                    Spacer().frame(width: 12)
                }
                Text(
                    "Pavan Kumar Anasosalu Vasu, Hadi Pour Ansari, Fartash Faghri, Raviteja Vemulapalli, Oncel Tuzel."
                )
                .padding(.horizontal, 12)

            }
            .foregroundColor(.secondary)
            .font(.system(size: 12))

            Spacer()
        }
        .textSelection(.enabled)
        .font(.system(size: 16))
        .padding(.bottom)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - View Modifiers
    struct AboutPanel: ViewModifier {
        func body(content: Content) -> some View {
            content
                .foregroundColor(.secondary)
                .padding(.horizontal, 15)
                .padding(.vertical, 30)
                .textSelection(.enabled)
        }
    }
}
