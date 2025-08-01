//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

/// Displays the current prompt formatted with an arrow
struct PromptPreview: View {

    let prompt: Prompt

    var body: some View {
        HStack {
            Image(systemName: "arrow.forward")
            Text("\(prompt.prefix) ") + Text("CLASSNAME").underline() + Text(" \(prompt.suffix)")
        }
        .lineLimit(1)
        .padding(.bottom, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 4)
    }
}

#Preview {
    VStack(alignment: .leading) {
        PromptPreview(prompt: .init(prefix: "A photo of", suffix: "", classNames: []))

        PromptPreview(
            prompt: .init(
                prefix: "A photo of", suffix: "with some longer thing at the end", classNames: []))

    }
    .padding()
}
