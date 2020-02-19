import Foundation
import TensorFlow
import GANUtils

@differentiable
func lrelu<Scalar: TensorFlowFloatingPoint>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
    leakyRelu(x, alpha: 0.2)
}

public func sampleNoise(size: Int, latentSize: Int) -> Tensor<Float> {
    Tensor(randomNormal: [size, latentSize])
}
