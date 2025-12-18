//
//  CoreMLGainFieldExecutor.swift
//  ProjectTeal
//
//  Wraps the gain-field Core ML model with FP16-first configuration and
//  predictable I/O for tiled inference. Converts RGBImage tiles and guidance
//  into MLMultiArray input, reads gain/detail outputs, upsamples the gain
//  field, and composes the linear 48MP prediction. INT8 models can be loaded
//  after visual sign-off by switching the precision flag.
//

import CoreML
import Foundation
import Metal
import simd

protocol GainFieldModelPerforming {
    func prediction(from input: MLFeatureProvider) throws -> MLFeatureProvider
}

extension MLModel: GainFieldModelPerforming {}

/// Precision options for the gain-field Core ML model.
enum GainFieldModelPrecision {
    case fp16
    case int8Candidate
}

/// Common I/O naming for the gain-field model.
struct GainFieldModelIO {
    let inputName: String
    let gainOutputName: String
    let detailOutputName: String

    init(inputName: String = "input", gainOutputName: String = "gain", detailOutputName: String = "detail") {
        self.inputName = inputName
        self.gainOutputName = gainOutputName
        self.detailOutputName = detailOutputName
    }
}

/// Prediction artifacts returned by the gain-field Core ML executor.
struct GainFieldPredictionResult {
    let gain: GainFieldTile
    let detail: RGBImage
    let composed: RGBImage
}

/// Executes gain-field Core ML inference with FP16-first configuration.
final class CoreMLGainFieldExecutor {
    private let model: GainFieldModelPerforming
    private let io: GainFieldModelIO
    private let precision: GainFieldModelPrecision

    init(modelURL: URL,
         io: GainFieldModelIO = GainFieldModelIO(),
         precision: GainFieldModelPrecision = .fp16,
         computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
         metalDevice: MTLDevice? = MTLCreateSystemDefaultDevice()) throws {
        let configuration = CoreMLGainFieldExecutor.makeConfiguration(precision: precision,
                                                                      computeUnits: computeUnits,
                                                                      metalDevice: metalDevice)
        self.model = try MLModel(contentsOf: modelURL, configuration: configuration)
        self.io = io
        self.precision = precision
    }

    init(model: GainFieldModelPerforming,
         io: GainFieldModelIO = GainFieldModelIO(),
         precision: GainFieldModelPrecision = .fp16) {
        self.model = model
        self.io = io
        self.precision = precision
    }

    /// Runs a single-tile prediction given a ProRAW48 tile and RAW12 guidance tile.
    func predict(proRaw48: RGBImage, guidance: RGBImage) throws -> GainFieldPredictionResult {
        guard proRaw48.width == guidance.width && proRaw48.height == guidance.height else {
            throw PredictionError.dimensionMismatch
        }

        let input = try makeInput(proRaw48: proRaw48, guidance: guidance)
        let provider = try MLDictionaryFeatureProvider(dictionary: [io.inputName: input])
        let output = try model.prediction(from: provider)

        guard let gainArray = output.featureValue(for: io.gainOutputName)?.multiArrayValue else {
            throw PredictionError.missingOutput(io.gainOutputName)
        }
        guard let detailArray = output.featureValue(for: io.detailOutputName)?.multiArrayValue else {
            throw PredictionError.missingOutput(io.detailOutputName)
        }

        let gainTile = try makeGainField(from: gainArray)
        let detailImage = try makeRGBImage(from: detailArray)
        let composed = compose(proRaw48: proRaw48, gain: gainTile, detail: detailImage)

        return GainFieldPredictionResult(gain: gainTile, detail: detailImage, composed: composed)
    }
}

private extension CoreMLGainFieldExecutor {
    enum PredictionError: Swift.Error {
        case dimensionMismatch
        case unsupportedDataType(MLMultiArrayDataType)
        case missingOutput(String)
        case unexpectedShape
    }

    static func makeConfiguration(precision: GainFieldModelPrecision,
                                  computeUnits: MLComputeUnits,
                                  metalDevice: MTLDevice?) -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        configuration.allowLowPrecisionAccumulationOnGPU = true
        configuration.preferredMetalDevice = metalDevice

        if precision == .int8Candidate {
            configuration.allowLowPrecisionAccumulationOnGPU = true
        }

        return configuration
    }

    func makeInput(proRaw48: RGBImage, guidance: RGBImage) throws -> MLMultiArray {
        let shape: [NSNumber] = [1, 6, proRaw48.height as NSNumber, proRaw48.width as NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        let strideB = array.strides[0].intValue
        let strideC = array.strides[1].intValue
        let strideY = array.strides[2].intValue
        let strideX = array.strides[3].intValue

        let pointer = UnsafeMutablePointer<Float>(OpaquePointer(array.dataPointer))
        let count = array.count
        pointer.assign(repeating: 0, count: count)

        for y in 0 ..< proRaw48.height {
            for x in 0 ..< proRaw48.width {
                let index = y * proRaw48.width + x
                let proPixel = proRaw48.pixels[index]
                let guidePixel = guidance.pixels[index]

                write(pixel: proPixel, to: pointer, channelOffset: 0, x: x, y: y, strideB: strideB, strideC: strideC, strideY: strideY, strideX: strideX)
                write(pixel: guidePixel, to: pointer, channelOffset: 3, x: x, y: y, strideB: strideB, strideC: strideC, strideY: strideY, strideX: strideX)
            }
        }

        return array
    }

    func write(pixel: SIMD3<Float>,
               to pointer: UnsafeMutablePointer<Float>,
               channelOffset: Int,
               x: Int,
               y: Int,
               strideB: Int,
               strideC: Int,
               strideY: Int,
               strideX: Int) {
        let batchIndex = 0
        let base = batchIndex * strideB + y * strideY + x * strideX
        pointer[base + (channelOffset + 0) * strideC] = pixel.x
        pointer[base + (channelOffset + 1) * strideC] = pixel.y
        pointer[base + (channelOffset + 2) * strideC] = pixel.z
    }

    func makeGainField(from array: MLMultiArray) throws -> GainFieldTile {
        guard array.shape.count == 4,
              array.shape[0].intValue == 1,
              array.shape[1].intValue == 3 else {
            throw PredictionError.unexpectedShape
        }

        let height = array.shape[2].intValue
        let width = array.shape[3].intValue
        let pixels = try readPixels(from: array, width: width, height: height)
        return GainFieldTile(width: width, height: height, pixels: pixels)
    }

    func makeRGBImage(from array: MLMultiArray) throws -> RGBImage {
        guard array.shape.count == 4,
              array.shape[0].intValue == 1,
              array.shape[1].intValue == 3 else {
            throw PredictionError.unexpectedShape
        }

        let height = array.shape[2].intValue
        let width = array.shape[3].intValue
        let pixels = try readPixels(from: array, width: width, height: height)
        return RGBImage(width: width, height: height, pixels: pixels)
    }

    func readPixels(from array: MLMultiArray, width: Int, height: Int) throws -> [SIMD3<Float>] {
        let strideB = array.strides[0].intValue
        let strideC = array.strides[1].intValue
        let strideY = array.strides[2].intValue
        let strideX = array.strides[3].intValue

        var pixels = [SIMD3<Float>](repeating: .zero, count: width * height)
        switch array.dataType {
        case .float32:
            let pointer = UnsafePointer<Float>(OpaquePointer(array.dataPointer))
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let base = y * strideY + x * strideX
                    let r = pointer[base + 0 * strideC]
                    let g = pointer[base + 1 * strideC]
                    let b = pointer[base + 2 * strideC]
                    pixels[y * width + x] = SIMD3<Float>(r, g, b)
                }
            }
        case .float16:
            let pointer = UnsafePointer<UInt16>(OpaquePointer(array.dataPointer))
            for y in 0 ..< height {
                for x in 0 ..< width {
                    let base = y * strideY + x * strideX
                    let r = Float(Float16(bitPattern: pointer[base + 0 * strideC]))
                    let g = Float(Float16(bitPattern: pointer[base + 1 * strideC]))
                    let b = Float(Float16(bitPattern: pointer[base + 2 * strideC]))
                    pixels[y * width + x] = SIMD3<Float>(r, g, b)
                }
            }
        default:
            throw PredictionError.unsupportedDataType(array.dataType)
        }

        return pixels
    }

    func compose(proRaw48: RGBImage, gain: GainFieldTile, detail: RGBImage) -> RGBImage {
        let upsampledGain = upsampleBilinear(gain: gain,
                                             targetWidth: proRaw48.width,
                                             targetHeight: proRaw48.height)
        var output = [SIMD3<Float>](repeating: .zero, count: proRaw48.width * proRaw48.height)

        for index in 0 ..< output.count {
            let combined = proRaw48.pixels[index] * upsampledGain[index] + detail.pixels[index]
            output[index] = simd.max(combined, SIMD3<Float>(repeating: 0))
        }

        return RGBImage(width: proRaw48.width, height: proRaw48.height, pixels: output)
    }

    func upsampleBilinear(gain: GainFieldTile, targetWidth: Int, targetHeight: Int) -> [SIMD3<Float>] {
        if gain.width == targetWidth && gain.height == targetHeight {
            return gain.pixels
        }

        var output = [SIMD3<Float>](repeating: .zero, count: targetWidth * targetHeight)
        let scaleX = Float(gain.width - 1) / Float(max(targetWidth - 1, 1))
        let scaleY = Float(gain.height - 1) / Float(max(targetHeight - 1, 1))

        for y in 0 ..< targetHeight {
            let srcY = Float(y) * scaleY
            let y0 = Int(floor(srcY))
            let y1 = min(y0 + 1, gain.height - 1)
            let ty = srcY - Float(y0)

            for x in 0 ..< targetWidth {
                let srcX = Float(x) * scaleX
                let x0 = Int(floor(srcX))
                let x1 = min(x0 + 1, gain.width - 1)
                let tx = srcX - Float(x0)

                let topLeft = gain[x0, y0]
                let topRight = gain[x1, y0]
                let bottomLeft = gain[x0, y1]
                let bottomRight = gain[x1, y1]

                let top = topLeft * (1 - tx) + topRight * tx
                let bottom = bottomLeft * (1 - tx) + bottomRight * tx
                let interpolated = top * (1 - ty) + bottom * ty

                output[y * targetWidth + x] = interpolated
            }
        }

        return output
    }
}
