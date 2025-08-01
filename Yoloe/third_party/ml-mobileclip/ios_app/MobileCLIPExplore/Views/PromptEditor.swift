//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import Foundation
import SwiftUI

// MARK: - Field Views and Layout
/// The primary label of a field
private struct FieldTitle: View {

    let title: String

    var body: some View {
        Text(title.uppercased())
            .font(.caption)
            .bold()
    }
}

/// A field with primary and secondary labels, e.g. shown when editing the `start of a prompt`
private struct FieldView<V: View>: View {

    let title: String
    let help: String
    let contents: V

    internal init(title: String, help: String, @ViewBuilder contents: () -> V) {
        self.title = title
        self.help = help
        self.contents = contents()
    }

    var body: some View {
        VStack(alignment: .leading) {
            FieldTitle(title: title)
                .padding(.horizontal, 10)

            contents
                .padding()
                .background(Color("PromptEditor"))
                .clipShape(RoundedRectangle(cornerRadius: 8))

            Text(help)
                .modifier(SecondaryLabel())
                .padding(.horizontal, 10)
        }
        .padding(.bottom, 15)
    }

}

// MARK: - Configure Prompt View
/// Displays a preview, instructions and fields needed to configure a prompt
struct PromptEditor: View {

    @Binding var prompt: Prompt

    var body: some View {
        VStack {
            ScrollView {
                Group {
                    PromptPreview(prompt: prompt)

                    Divider()

                    Text(
                        """
                        Configure the start and end of the prompt. A preview of the fully composed prompt for the text encoder is shown above. To customize class names, click the Classname Selection button.
                        """.replacingOccurrences(of: "\n", with: " ")
                    )
                    .modifier(SecondaryLabel())
                    .padding(.bottom, 30)
                }

                VStack(spacing: 8) {
                    FieldView(
                        title: "Start of prompt",
                        help: "The start of the prompt. Can be left blank."
                    ) {
                        TextField("No prefix", text: $prompt.prefix)
                    }

                    FieldView(
                        title: "Classnames",
                        help: "Classnames that will be scored."
                    ) {

                        // Classname Selection Button
                        NavigationLink {
                            ClassnamesEditor(classnames: $prompt.classNames)

                        } label: {
                            HStack {
                                if !prompt.classNames.isEmpty {

                                    // Display added classnames
                                    Text(prompt.classNames.joined(separator: ", "))
                                        .lineLimit(1)

                                } else {
                                    // Label shown if no classnames have been added
                                    Text("Classname Selection")

                                }
                                Spacer()
                                Image(systemName: "pencil")
                                    .padding(.vertical)
                            }
                        }
                    }

                    FieldView(
                        title: "End of prompt",
                        help: "The end of the prompt. Can be left blank."
                    ) {
                        TextField("No suffix", text: $prompt.suffix)
                    }
                }
                Spacer()
            }
        }
        .navigationTitle("Configure Prompt")
        .navigationBarTitleDisplayMode(.large)
        .padding(EdgeInsets(top: 10, leading: 15, bottom: 10, trailing: 15))
        .background(Color("PromptEditorBackground"))
    }
}

// MARK: - Manage Classnames View
/// Displays the current Classnames, and instructs the user how to add or remove them
struct ClassnamesEditor: View {

    @Binding var classnames: [String]

    @State private var className = ""

    var body: some View {
        VStack(alignment: .leading) {
            HStack {
                TextField("Add a new classname", text: $className)
                    .textInputAutocapitalization(.never)
                    .onSubmit(addClassName)
                    .padding()
                    .background(Color("PromptEditor"))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                Button(action: { addClassName() }) {
                    Image(systemName: "plus.circle")
                        .foregroundStyle(Color("AddClassName"))
                        .padding()
                }
            }
            .padding(.bottom, 16)

            Divider()

            Text("Add the classnames that you would like the app to detect.")
                .modifier(SecondaryLabel())
                .padding(.bottom, 16)

            // Display classnames as they are added
            if !classnames.isEmpty {
                Group {
                    FieldTitle(title: "Current Classnames")

                    Text("Swipe left to remove items.")
                        .modifier(SecondaryLabel())
                }
                .padding(.horizontal, 10)

                List {
                    ForEach(classnames, id: \.self) { className in

                        // A classname in a row
                        HStack {
                            Text(className)
                            Spacer()
                        }
                        .listRowBackground(Color("PromptEditor"))
                    }
                    .onDelete { indexes in
                        classnames.remove(atOffsets: indexes)
                    }
                }
                .ignoresSafeArea(.keyboard)
                .environment(\.defaultMinListRowHeight, 25)
                .scrollContentBackground(.hidden)
            } else {
                Spacer()
            }

        }
        .onAppear {
            classnames.sort { $0.lowercased() < $1.lowercased() }
        }
        .navigationTitle("Manage Classnames")
        .navigationBarTitleDisplayMode(.large)
        .padding()
        .background(Color("PromptEditorBackground"))
    }

    private func addClassName() {
        let nameToAdd = className.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !nameToAdd.isEmpty else { return }
        guard !classnames.contains(nameToAdd) else { return }

        withAnimation {
            classnames.append(nameToAdd)
            classnames.sort { $0.lowercased() < $1.lowercased() }

            className = ""
        }
    }
}

// MARK: - View Modifiers
struct SecondaryLabel: ViewModifier {
    func body(content: Content) -> some View {
        content
            .foregroundColor(.secondary)
            .font(.caption)
    }
}

// MARK: - Previews
#Preview {
    VStack {
        FieldView(
            title: "Start of prompt",
            help: "The start of the prompt.  Can be left blank."
        ) {
            TextField("", text: .constant("A photo of"))
        }

        ClassnamesEditor(classnames: .constant(["cat", "dog", "chicken"]))
    }
    .padding()
    .background(Color(UIColor.systemGroupedBackground))
}
