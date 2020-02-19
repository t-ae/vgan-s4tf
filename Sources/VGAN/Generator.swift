import Foundation
import TensorFlow
import GANUtils

struct GBlock: Layer {
    var conv1: TransposedConv2D<Float>
    var conv2: Conv2D<Float>
    var toRGB: Conv2D<Float>
    
    init(
        inputChannels: Int,
        outputChannels: Int
    ) {
        conv1 = TransposedConv2D(filterShape: (4, 4, outputChannels, inputChannels),
                                 strides: (2, 2),
                                 padding: .same,
                                 activation: lrelu,
                                 filterInitializer: heNormal())
        conv2 = Conv2D(filterShape: (3, 3, outputChannels, outputChannels),
                       padding: .same,
                       activation: lrelu,
                       filterInitializer: heNormal())
        toRGB = Conv2D(filterShape: (1, 1, outputChannels, 3),
                       filterInitializer: heNormal())
    }
    
    struct Output: Differentiable {
        var features: Tensor<Float>
        var images: Tensor<Float>
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Output {
        var x = input
        
        x = conv1(x)
        x = conv2(x)
        let images = toRGB(x)
        
        return Output(features: x, images: images)
    }
}

struct Generator: Layer {
    struct Config: Codable {
        var latentSize: Int
        var resizeMethod: Resize.Method
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var head: Dense<Float>
    var x4Block: GBlock
    var x8Block: GBlock
    var x16Block: GBlock
    var x32Block: GBlock
    var x64Block: GBlock
    var x128Block: GBlock
    var x256Block: GBlock
    
    var resize2x: Resize
    
    @noDerivative
    let imageSize: ImageSize
    
    init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        
        let baseChannels = config.baseChannels
        let maxChannels = config.maxChannels
        
        func ioChannels(for size: ImageSize) -> (i: Int, o: Int) {
            guard size <= imageSize else {
                return (0, 0)
            }
            let d = imageSize.log2 - size.log2
            let o = baseChannels * 1 << d
            let i = o * 2
            return (min(i, maxChannels), min(o, maxChannels))
        }
        
        let io4 = ioChannels(for: .x4)
        head = Dense(inputSize: config.latentSize, outputSize: io4.i * 2 * 2, activation: lrelu)
        x4Block = GBlock(inputChannels: io4.i, outputChannels: io4.o)
        
        let io8 = ioChannels(for: .x8)
        x8Block = GBlock(inputChannels: io8.i, outputChannels: io8.o)
        
        let io16 = ioChannels(for: .x16)
        x16Block = GBlock(inputChannels: io16.i, outputChannels: io16.o)
        
        let io32 = ioChannels(for: .x32)
        x32Block = GBlock(inputChannels: io32.i, outputChannels: io32.o)
        
        let io64 = ioChannels(for: .x64)
        x64Block = GBlock(inputChannels: io64.i, outputChannels: io64.o)
        
        let io128 = ioChannels(for: .x128)
        x128Block = GBlock(inputChannels: io128.i, outputChannels: io128.o)
        
        let io256 = ioChannels(for: .x256)
        x256Block = GBlock(inputChannels: io256.i, outputChannels: io256.o)
        
        resize2x = Resize(config.resizeMethod, outputSize: .factor(x: 2, y: 2), alignCorners: true)
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = head(x).reshaped(to: [input.shape[0], 2, 2, -1]) // 2x2
        
        let x4 = x4Block(x)
        var images = x4.images
        if imageSize == .x4 {
            return images
        }
        
        let x8 = x8Block(x4.features)
        images = x8.images + resize2x(images)
        if imageSize == .x8 {
            return images / 2 // average
        }
        
        let x16 = x16Block(x8.features)
        images = x16.images + resize2x(images)
        if imageSize == .x16 {
            return images / 3
        }
        
        let x32 = x32Block(x16.features)
        images = x32.images + resize2x(images)
        if imageSize == .x32 {
            return images / 4
        }
        
        let x64 = x64Block(x32.features)
        images = x64.images + resize2x(images)
        if imageSize == .x64 {
            return images / 5
        }
        
        let x128 = x128Block(x64.features)
        images = x128.images + resize2x(images)
        if imageSize == .x128 {
            return images / 6
        }
        
        let x256 = x256Block(x128.features)
        images = x256.images + resize2x(images)
        return images / 7
    }
}
