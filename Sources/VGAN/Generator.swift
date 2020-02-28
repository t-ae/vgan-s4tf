import Foundation
import TensorFlow
import GANUtils

struct Blur2D: ParameterlessLayer {
    @noDerivative
    let filter: Tensor<Float>
    
    init(channels: Int) {
        let f = Tensor<Float>([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
            .reshaped(to: [3, 3, 1, 1]) / 16
        filter = f.tiled(multiples: Tensor([1, 1, Int32(channels), 1]))
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        depthwiseConv2D(input, filter: filter, strides: (1, 1, 1, 1), padding: .same)
    }
}

struct GBlock: Layer {
    var conv1: TransposedConv2D<Float>
    var conv2: TransposedConv2D<Float>
    var shortcut: Conv2D<Float>
    
    @noDerivative
    let blur: Blur2D
    
    @noDerivative
    let learnableSC: Bool
    
    var bn1: BatchNorm<Float>
    var bn2: BatchNorm<Float>
    
    var resize2x: Resize
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        resize2x: Resize
    ) {
        conv1 = TransposedConv2D(filterShape: (4, 4, outputChannels, inputChannels),
                                 strides: (2, 2),
                                 padding: .same,
                                 filterInitializer: heNormal())
        blur = Blur2D(channels: outputChannels)
        conv2 = TransposedConv2D(filterShape: (3, 3, outputChannels, outputChannels),
                                 padding: .same,
                                 filterInitializer: heNormal())
        
        learnableSC = inputChannels != outputChannels
        shortcut = Conv2D(filterShape: (1, 1, learnableSC ? inputChannels: 0, outputChannels),
                          filterInitializer: heNormal())
        
        bn1 = BatchNorm(featureCount: inputChannels)
        bn2 = BatchNorm(featureCount: outputChannels)
        self.resize2x = resize2x
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(leakyRelu(bn1(x)))
        x = blur(x)
        x = conv2(leakyRelu(bn2(x)))
        
        var sc = resize2x(input)
        if learnableSC {
            sc = shortcut(sc)
        }
        return 0.1*x + sc
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
    
    var toRGB: Conv2D<Float>
    
    @noDerivative
    let imageSize: ImageSize
    
    init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        
        let baseChannels = config.baseChannels
        let maxChannels = config.maxChannels
        let resize = Resize(config.resizeMethod, outputSize: .factor(x: 2, y: 2), alignCorners: true)
        
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
        head = Dense(inputSize: config.latentSize, outputSize: io4.i * 2 * 2)
        x4Block = GBlock(inputChannels: io4.i, outputChannels: io4.o, resize2x: resize)
        
        let io8 = ioChannels(for: .x8)
        x8Block = GBlock(inputChannels: io8.i, outputChannels: io8.o, resize2x: resize)
        
        let io16 = ioChannels(for: .x16)
        x16Block = GBlock(inputChannels: io16.i, outputChannels: io16.o, resize2x: resize)
        
        let io32 = ioChannels(for: .x32)
        x32Block = GBlock(inputChannels: io32.i, outputChannels: io32.o, resize2x: resize)
        
        let io64 = ioChannels(for: .x64)
        x64Block = GBlock(inputChannels: io64.i, outputChannels: io64.o, resize2x: resize)
        
        let io128 = ioChannels(for: .x128)
        x128Block = GBlock(inputChannels: io128.i, outputChannels: io128.o, resize2x: resize)
        
        let io256 = ioChannels(for: .x256)
        x256Block = GBlock(inputChannels: io256.i, outputChannels: io256.o, resize2x: resize)
        
        toRGB = Conv2D(filterShape: (1, 1, baseChannels, 3),
                       activation: tanh,
                       filterInitializer: heNormal())
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        
        x = head(x).reshaped(to: [input.shape[0], 2, 2, -1]) // 2x2
        
        x = x4Block(x)
        if imageSize == .x4 {
            return toRGB(leakyRelu(x))
        }
        
        x = x8Block(x)
        if imageSize == .x8 {
            return toRGB(leakyRelu(x))
        }
        
        x = x16Block(x)
        if imageSize == .x16 {
            return toRGB(leakyRelu(x))
        }
        
        x = x32Block(x)
        if imageSize == .x32 {
            return toRGB(leakyRelu(x))
        }
        
        x = x64Block(x)
        if imageSize == .x64 {
            return toRGB(leakyRelu(x))
        }
        
        x = x128Block(x)
        if imageSize == .x128 {
            return toRGB(leakyRelu(x))
        }
        
        x = x256Block(x)
        return toRGB(leakyRelu(x))
    }
}

extension Generator {
    func inferring(from input: Tensor<Float>, batchSize: Int) -> Tensor<Float> {
        let x = input.reshaped(to: [-1, batchSize, input.shape[1]])
        return Tensor(concatenating: (0..<x.shape[0]).map { inferring(from: x[$0]) },
                      alongAxis: 0)
    }
}
