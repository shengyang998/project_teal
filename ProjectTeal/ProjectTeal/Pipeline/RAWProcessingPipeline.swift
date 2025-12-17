//
//  RAWProcessingPipeline.swift
//  ProjectTeal
//
//  Baseline RAW ingest and linear DNG export pipeline used to validate 48MP processing steps.
//

import Foundation
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

/// Encapsulates ingestion of AVCapturePhoto RAW data into a linear RGB DNG artifact.
final class RAWProcessingPipeline {
    private let contextProvider: CIContextProvider
    private let dngWriter: LinearDNGWriter
    private let processingQueue: DispatchQueue

    struct Result {
        let dng: LinearDNGWriter.Output
        let metadata: RAWMetadata
    }

    enum Error: Swift.Error {
        case missingFileData
        case unsupportedImageSource
        case renderFailure
    }

    init(contextProvider: CIContextProvider = .shared,
         dngWriter: LinearDNGWriter = LinearDNGWriter(),
         processingQueue: DispatchQueue = DispatchQueue(label: "ProjectTeal.raw.pipeline")) {
        self.contextProvider = contextProvider
        self.dngWriter = dngWriter
        self.processingQueue = processingQueue
    }

    /// Executes the baseline pipeline: load RAW -> ensure linear space -> export DNG.
    /// - Parameters:
    ///   - photo: Captured photo (ideally Apple ProRAW or Bayer RAW).
    ///   - completion: Async callback delivered on the main queue.
    func process(photo: AVCapturePhoto, completion: @escaping (Swift.Result<Result, Error>) -> Void) {
        processingQueue.async { [weak self] in
            guard let self else { return }
            let result = self.processInternal(photo: photo)
            DispatchQueue.main.async {
                completion(result)
            }
        }
    }

    private func processInternal(photo: AVCapturePhoto) -> Swift.Result<Result, Error> {
        guard let data = photo.fileDataRepresentation() else {
            return .failure(.missingFileData)
        }

        guard let imageSource = CGImageSourceCreateWithData(data as CFData, nil) else {
            return .failure(.unsupportedImageSource)
        }

        let metadata = RAWMetadata(imageSource: imageSource)
        guard let cgImage = makeLinearCGImage(from: data) else {
            return .failure(.renderFailure)
        }

        let outputURL = RAWProcessingPipeline.makeOutputURL()
        do {
            let dngOutput = try dngWriter.write(cgImage: cgImage,
                                                metadata: CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
                                                destinationURL: outputURL)
            return .success(Result(dng: dngOutput, metadata: metadata))
        } catch {
            return .failure(.renderFailure)
        }
    }

    private func makeLinearCGImage(from data: Data) -> CGImage? {
        // Prefer the RAW image at index 0; CIImage will honor orientation metadata.
        let options: [CIImageOption: Any] = [
            .applyOrientationProperty: true,
            .cache: false
        ]
        guard let ciImage = CIImage(data: data, options: options) else {
            return nil
        }

        // Force linear output; remove any tone curve by transforming from sRGB tone curve to linear.
        let linearImage = ciImage.applyingFilter("CISRGBToneCurveToLinear")
        return contextProvider.renderLinearCGImage(from: linearImage)
    }

    private static func makeOutputURL() -> URL {
        let filename = "capture-linear-\(Int(Date().timeIntervalSince1970)).dng"
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ProjectTeal", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appendingPathComponent(filename)
    }
}
