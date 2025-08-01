//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import CoreML
import SwiftUI

struct ContentView: View {
    @State private var camera = CameraController()

    var body: some View {
        InferenceView(
            camera: camera,
            backCamera: $camera.backCamera
        )
        .ignoresSafeArea(edges: [.bottom])
        .task {
            camera.start()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
