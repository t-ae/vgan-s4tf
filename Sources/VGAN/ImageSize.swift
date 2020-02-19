import Foundation

public enum ImageSize: Int, Codable, CaseIterable {
    case x4 = 4
    case x8 = 8
    case x16 = 16
    case x32 = 32
    case x64 = 64
    case x128 = 128
    case x256 = 256
    
    public var log2: Int {
        return Int(Foundation.log2(Float(rawValue)))
    }
    
    public var name: String {
        "\(rawValue)x\(rawValue)"
    }
}

extension ImageSize: Comparable {
    public static func < (lhs: ImageSize, rhs: ImageSize) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
