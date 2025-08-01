//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2024 Apple Inc. All Rights Reserved.
//

import AVFoundation
import CoreImage
import UIKit

@Observable
class CameraController: NSObject {

    private var framesContinuation: AsyncStream<CMSampleBuffer>.Continuation?

    public var backCamera = true {
        didSet {
            stop()
            start()
        }
    }

    private var permissionGranted = true
    private var captureSession: AVCaptureSession?
    private let sessionQueue = DispatchQueue(label: "sessionQueue")

    public func attach(continuation: AsyncStream<CMSampleBuffer>.Continuation) {
        sessionQueue.async {
            self.framesContinuation = continuation
        }
    }

    public func detatch() {
        sessionQueue.async {
            self.framesContinuation = nil
        }
    }

    public func stop() {
        sessionQueue.sync { [self] in
            captureSession?.stopRunning()
            captureSession = nil
        }

    }

    public func start() {
        sessionQueue.async { [self] in
            let captureSession = AVCaptureSession()
            self.captureSession = captureSession

            self.checkPermission()
            self.setupCaptureSession(position: backCamera ? .back : .front)
            captureSession.startRunning()
        }
    }

    private func setOrientation(_ orientation: UIDeviceOrientation) {
        guard let captureSession else { return }

        let angle: Double?
        switch orientation {
        case .unknown, .faceDown:
            angle = nil
        case .portrait, .faceUp:
            angle = 90
        case .portraitUpsideDown:
            angle = 270
        case .landscapeLeft:
            angle = 0
        case .landscapeRight:
            angle = 180
        @unknown default:
            angle = nil
        }

        if let angle {
            for output in captureSession.outputs {
                output.connection(with: .video)?.videoRotationAngle = angle
            }
        }
    }

    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            // The user has previously granted access to the camera.
            self.permissionGranted = true

        case .notDetermined:
            // The user has not yet been asked for camera access.
            self.requestPermission()

        // Combine the two other cases into the default case
        default:
            self.permissionGranted = false
        }
    }

    func requestPermission() {
        // Strong reference not a problem here but might become one in the future.
        AVCaptureDevice.requestAccess(for: .video) { [unowned self] granted in
            self.permissionGranted = granted
        }
    }

    func setupCaptureSession(position: AVCaptureDevice.Position) {
        guard let captureSession else { return }

        let videoOutput = AVCaptureVideoDataOutput()

        guard permissionGranted else {
            print("No permission for camera")
            return
        }

        let videoDeviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInDualCamera, .builtInWideAngleCamera],
            mediaType: .video,
            position: position)

        guard
            let videoDevice = videoDeviceDiscoverySession.devices.first
        else {
            print("Unable to find video device")
            return
        }
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else {
            print("Unable to create AVCaptureDeviceInput")
            return
        }
        guard captureSession.canAddInput(videoDeviceInput) else {
            print("Unable to add input")
            return
        }
        captureSession.addInput(videoDeviceInput)

        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sampleBufferQueue"))
        captureSession.addOutput(videoOutput)
        captureSession.sessionPreset = AVCaptureSession.Preset.vga640x480

        if videoDevice.isContinuityCamera {
            setOrientation(.portrait)
        } else {
            setOrientation(UIDevice.current.orientation)
        }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        if sampleBuffer.isValid && sampleBuffer.imageBuffer != nil {
            framesContinuation?.yield(sampleBuffer)
        }
    }
}
