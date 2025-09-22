//
//  AppRootView.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import UIKit
import AVFoundation
import Photos

final class CameraViewController: UIViewController, AVCapturePhotoCaptureDelegate {
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "ProjectTeal.camera.session")

    private var photoOutput = AVCapturePhotoOutput()
    private var previewLayer: AVCaptureVideoPreviewLayer?
    private var rawPixelFormatType: OSType?

    private let shutterButton: UIButton = {
        let button = UIButton(type: .system)
        button.backgroundColor = .white
        button.layer.cornerRadius = 35
        button.layer.cornerCurve = .continuous
        button.layer.borderColor = UIColor.black.withAlphaComponent(0.25).cgColor
        button.layer.borderWidth = 2
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        configureUI()
        checkCameraAuthorization()
    }

    private func checkCameraAuthorization() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            configureSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                DispatchQueue.main.async {
                    if granted {
                        self.configureSession()
                    } else {
                        self.showPermissionAlert()
                    }
                }
            }
        default:
            showPermissionAlert()
        }
    }

    private func configureSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.captureSession.beginConfiguration()
            self.captureSession.sessionPreset = .photo

            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                assertionFailure("Back camera is unavailable.")
                self.captureSession.commitConfiguration()
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: camera)
                if self.captureSession.canAddInput(input) {
                    self.captureSession.addInput(input)
                }
            } catch {
                assertionFailure("Failed to create camera input: \(error)")
                self.captureSession.commitConfiguration()
                return
            }

            let output = self.photoOutput
            output.isHighResolutionCaptureEnabled = true
            if self.captureSession.canAddOutput(output) {
                self.captureSession.addOutput(output)
            }

            // Enable Apple ProRAW if a supported RAW pixel format is available
            let availableRaw = output.availableRawPhotoPixelFormatTypes
            if let preferredAppleProRAW = self.preferredRawPixelFormat(from: availableRaw.filter({ AVCapturePhotoOutput.isAppleProRAWPixelFormat($0) })) {
                output.isAppleProRAWEnabled = true
                self.rawPixelFormatType = preferredAppleProRAW
            } else {
                self.rawPixelFormatType = nil
            }

            self.captureSession.commitConfiguration()

            DispatchQueue.main.async {
                self.configurePreviewOnMain()
            }

            self.captureSession.startRunning()
        }
    }

    private func preferredRawPixelFormat(from candidates: [OSType]) -> OSType? {
        // Prefer common 14-bit Bayer formats when available
        let preferred: [OSType] = [
            kCVPixelFormatType_14Bayer_RGGB,
            kCVPixelFormatType_14Bayer_GRBG,
            kCVPixelFormatType_14Bayer_GBRG,
            kCVPixelFormatType_14Bayer_BGGR
        ]
        for pf in preferred where candidates.contains(pf) { return pf }
        return candidates.first
    }

    private func configurePreviewOnMain() {
        let layer = AVCaptureVideoPreviewLayer(session: captureSession)
        layer.videoGravity = .resizeAspectFill
        view.layer.insertSublayer(layer, at: 0)
        previewLayer = layer
        view.setNeedsLayout()
    }

    private func configureUI() {
        view.addSubview(shutterButton)
        NSLayoutConstraint.activate([
            shutterButton.widthAnchor.constraint(equalToConstant: 70),
            shutterButton.heightAnchor.constraint(equalToConstant: 70),
            shutterButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            shutterButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24)
        ])
        shutterButton.addTarget(self, action: #selector(capturePhoto), for: .touchUpInside)
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer?.frame = view.bounds
    }

    @objc private func capturePhoto() {
        var settings: AVCapturePhotoSettings
        if let rawType = rawPixelFormatType, photoOutput.isAppleProRAWEnabled {
            settings = AVCapturePhotoSettings(rawPixelFormatType: rawType)
            settings.isHighResolutionPhotoEnabled = true
        } else {
            if photoOutput.availablePhotoCodecTypes.contains(.hevc) {
                settings = AVCapturePhotoSettings(format: [AVVideoCodecKey: AVVideoCodecType.hevc])
            } else {
                settings = AVCapturePhotoSettings()
            }
            settings.isHighResolutionPhotoEnabled = true
        }
        photoOutput.capturePhoto(with: settings, delegate: self)
    }

    // MARK: - AVCapturePhotoCaptureDelegate

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error {
            // For development: programmer-visible failure
            assertionFailure("Capture failed: \(error)")
            return
        }

        guard let data = photo.fileDataRepresentation() else {
            assertionFailure("fileDataRepresentation() returned nil")
            return
        }

        saveToPhotoLibrary(data: data)
    }

    private func saveToPhotoLibrary(data: Data) {
        PHPhotoLibrary.requestAuthorization { status in
            guard status == .authorized || status == .limited else {
                return
            }
            PHPhotoLibrary.shared().performChanges({
                let creationRequest = PHAssetCreationRequest.forAsset()
                let options = PHAssetResourceCreationOptions()
                creationRequest.addResource(with: .photo, data: data, options: options)
            }, completionHandler: nil)
        }
    }

    private func showPermissionAlert() {
        let alert = UIAlertController(title: "Camera Access Needed",
                                      message: "Enable camera access in Settings to capture photos.",
                                      preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}


