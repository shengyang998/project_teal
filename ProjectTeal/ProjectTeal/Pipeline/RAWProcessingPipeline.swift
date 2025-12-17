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
    private let contextProvider: CIContextProviding
    private let dngWriter: LinearDNGWriter
    private let processingQueue: DispatchQueue
    private let rawParser: RAW12Parser
    private let proRAWLinearizer: ProRAWLinearizer

    struct Result {
        let dng: LinearDNGWriter.Output
        let metadata: RAWMetadata
        let rawParsing: RAW12Parser.Parsed
    }

    enum Error: Swift.Error {
        case missingFileData
        case unsupportedImageSource
        case renderFailure
    }

    init(contextProvider: CIContextProviding = CIContextProvider.shared,
         dngWriter: LinearDNGWriter = LinearDNGWriter(),
         processingQueue: DispatchQueue = DispatchQueue(label: "ProjectTeal.raw.pipeline"),
         rawParser: RAW12Parser = RAW12Parser(),
         proRAWLinearizer: ProRAWLinearizer? = nil) {
        self.contextProvider = contextProvider
        self.dngWriter = dngWriter
        self.processingQueue = processingQueue
        self.rawParser = rawParser
        self.proRAWLinearizer = proRAWLinearizer ?? ProRAWLinearizer(contextProvider: contextProvider)
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
        guard let parsed = rawParser.parse(imageSource: imageSource) else {
            return .failure(.renderFailure)
        }

        guard let cgImage = proRAWLinearizer.makeLinearCGImage(from: data,
                                                                normalization: parsed.normalization) else {
            return .failure(.renderFailure)
        }

        let outputURL = RAWProcessingPipeline.makeOutputURL()
        do {
            let dngOutput = try dngWriter.write(cgImage: cgImage,
                                                metadata: CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [CFString: Any],
                                                destinationURL: outputURL)
            return .success(Result(dng: dngOutput, metadata: metadata, rawParsing: parsed))
        } catch {
            return .failure(.renderFailure)
        }
    }

    private static func makeOutputURL() -> URL {
        let filename = "capture-linear-\(Int(Date().timeIntervalSince1970)).dng"
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ProjectTeal", isDirectory: true)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder.appendingPathComponent(filename)
    }
}
