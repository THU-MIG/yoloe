//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import CoreImage

extension CIImage {

    /// cropToSquare image
    /// - Returns: Cropped image
    func cropToSquare() -> CIImage? {
        let size = min(self.extent.width, self.extent.height)
        let x = round((self.extent.width - size) / 2)
        let y = round((self.extent.height - size) / 2)

        let cropRect = CGRect(
            x: x,
            y: y,
            width: size,
            height: size
        )

        let translate = CGAffineTransform(translationX: -x, y: -y)

        return
            self
            .cropped(to: cropRect)
            .transformed(by: translate)
    }

    /// Resize image
    /// - Parameter size: Size to resize to
    /// - Returns: Resized image
    func resize(size: CGSize) -> CIImage? {
        let scaleX = size.width / self.extent.width
        let scaleY = size.height / self.extent.height
        return self.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }
}
