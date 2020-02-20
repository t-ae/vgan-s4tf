import Foundation
import TensorFlow
import GANUtils

public func sampleNoise(size: Int, latentSize: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, latentSize])
}
