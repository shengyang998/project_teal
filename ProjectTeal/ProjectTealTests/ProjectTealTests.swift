//
//  ProjectTealTests.swift
//  ProjectTealTests
//
//  Created by Soleil Yu on 2025/9/23.
//

import CoreGraphics
@testable import ProjectTeal
import Testing

struct ProjectTealTests {
    @Test func parsesRAW12Metadata() throws {
        let rawDictionary: [CFString: Any] = [
            kCGImagePropertyRawBlackLevel: 64.0,
            kCGImagePropertyRawWhiteLevel: 4095.0,
            kCGImagePropertyExifWhiteBalanceBias: [2.0, 1.0, 1.5],
            kCGImagePropertyDNGCFAPattern: [0, 1, 1, 2]
        ]

        let parser = RAW12Parser()
        let parsed = parser.parse(rawProperties: rawDictionary, pixelSize: CGSize(width: 4000, height: 3000))

        #expect(parsed?.blackLevel == 64)
        #expect(parsed?.whiteLevel == 4095)
        #expect(parsed?.whiteBalanceGains == SIMD3(2, 1, 1.5))
        #expect(parsed?.cfaPattern == .rggb)
        #expect(parsed?.pixelSize == CGSize(width: 4000, height: 3000))
    }

    @Test func parsesDNGFallbackMetadata() throws {
        let rawDictionary: [CFString: Any] = [
            kCGImagePropertyDNGBlackLevel: [128.0, 128.0, 128.0, 128.0],
            kCGImagePropertyDNGWhiteLevel: 16383.0,
            kCGImagePropertyDNGWhiteBalance: [1.1, 1.0, 0.9]
        ]

        let parser = RAW12Parser()
        let parsed = parser.parse(rawProperties: rawDictionary, pixelSize: nil)

        #expect(parsed?.blackLevel == 128)
        #expect(parsed?.whiteLevel == 16383)
        #expect(parsed?.whiteBalanceGains == SIMD3(1.1, 1.0, 0.9))
        #expect(parsed?.cfaPattern == .unknown)
        #expect(parsed?.pixelSize == nil)
    }

    @Test func normalizesPixelValues() {
        let normalization = RAW12Parser.Normalization(blackLevel: 64, whiteLevel: 4095, whiteBalanceGains: SIMD3(1, 1, 1))

        #expect(normalization.normalize(value: 64) == 0)
        #expect(abs(normalization.normalize(value: 2048) - 0.5) < 1e-6)
        #expect(normalization.normalize(value: 5000) == 1)
    }

    @Test func stripsCFAMetadataAndAddsLinearFlags() {
        let writer = LinearDNGWriter()
        let metadata: [CFString: Any] = [
            kCGImagePropertyDNGDictionary: [
                kCGImagePropertyDNGCFAPattern: [0, 1, 1, 2],
                kCGImagePropertyDNGCFARepeatPatternDim: [2, 2]
            ],
            kCGImagePropertyTIFFDictionary: [
                kCGImagePropertyTIFFSamplesPerPixel: 1
            ]
        ]

        let properties = writer.makeDestinationProperties(metadata: metadata,
                                                          options: .init(tileSize: 512, compression: .losslessJPEG))

        let dng = properties[kCGImagePropertyDNGDictionary] as? [CFString: Any]
        #expect(dng?[kCGImagePropertyDNGCFAPattern] == nil)
        #expect(dng?[kCGImagePropertyDNGCFARepeatPatternDim] == nil)
        #expect(dng?[kCGImagePropertyDNGIsLinearRaw] as? Bool == true)

        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
        #expect(tiff?[kCGImagePropertyTIFFSamplesPerPixel] as? Int == 3)
        #expect(tiff?[kCGImagePropertyTIFFCompression] as? Int == LinearDNGWriter.Options.Compression.losslessJPEG.tiffValue)
        #expect(tiff?[kCGImagePropertyTIFFTileWidth] as? Int == 512)
        #expect(tiff?[kCGImagePropertyTIFFTileLength] as? Int == 512)
    }

    @Test func removesTileHintsWhenUnset() {
        let writer = LinearDNGWriter()
        let properties = writer.makeDestinationProperties(metadata: nil, options: .init())

        let tiff = properties[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
        #expect(tiff?[kCGImagePropertyTIFFTileWidth] == nil)
        #expect(tiff?[kCGImagePropertyTIFFTileLength] == nil)
        #expect(tiff?[kCGImagePropertyTIFFSamplesPerPixel] as? Int == 3)
    }
}
