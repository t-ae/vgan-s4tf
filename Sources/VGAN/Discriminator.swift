import Foundation
import TensorFlow
import GANUtils

struct DBlock: Layer {
    var conv1: SNConv2D<Float>
    var conv2: SNConv2D<Float>
    var shortcut: SNConv2D<Float>
    
    @noDerivative
    let learnableSC: Bool
    
    var avgPool = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    
    init(
        inputChannels: Int,
        outputChannels: Int,
        enableSpectralNormalization: Bool
    ) {
        conv1 = SNConv2D(filterShape: (3, 3, inputChannels, outputChannels),
                         padding: .same,
                         spectralNormalizationEnabled: enableSpectralNormalization,
                         filterInitializer: heNormal())
        conv2 = SNConv2D(filterShape: (4, 4, outputChannels, outputChannels),
                         strides: (2, 2),
                         padding: .same,
                         spectralNormalizationEnabled: enableSpectralNormalization,
                         filterInitializer: heNormal())
        
        learnableSC = inputChannels != outputChannels
        shortcut = SNConv2D(filterShape: (1, 1, inputChannels, learnableSC ? outputChannels : 0),
                            spectralNormalizationEnabled: enableSpectralNormalization,
                            filterInitializer: heNormal())
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        var x = input
        x = conv1(leakyRelu(x))
        x = conv2(leakyRelu(x))
        
        var sc = avgPool(input)
        if learnableSC {
            sc = shortcut(sc)
        }
        
        return 0.1*x + sc
    }
}

struct Discriminator: Layer {
    struct Config: Codable {
        var encodedSize: Int
        var enableSpectralNormalization: Bool
        var baseChannels: Int = 8
        var maxChannels: Int = 256
    }
    
    var x256Block: DBlock
    var x128Block: DBlock
    var x64Block: DBlock
    var x32Block: DBlock
    var x16Block: DBlock
    var x8Block: DBlock
//    var x4Block: DBlock
    
    var norm: InstanceNorm<Float>
    
    var meanConv: SNConv2D<Float>
    var logVarConv: SNConv2D<Float>
    
    var tail: SNConv2D<Float>
    
    var fromRGB: SNConv2D<Float>
    
    var avgPool: AvgPool2D<Float> = AvgPool2D(poolSize: (2, 2), strides: (2, 2))
    
    @noDerivative
    private let imageSize: ImageSize
    
    @noDerivative
    var reparametrize: Bool = false
    
    public init(config: Config, imageSize: ImageSize) {
        self.imageSize = imageSize
        let enableSN = config.enableSpectralNormalization
        
        func ioChannels(for size: ImageSize) -> (i: Int, o: Int) {
            guard size <= imageSize else {
                return (0, 0)
            }
            let d = imageSize.log2 - size.log2
            let i = config.baseChannels * 1 << d
            let o = i * 2
            return (min(i, config.maxChannels), min(o, config.maxChannels))
        }
        
        fromRGB = SNConv2D(filterShape: (1, 1, 3, config.baseChannels),
                           spectralNormalizationEnabled: enableSN,
                           filterInitializer: heNormal())
        
        let io256 = ioChannels(for: .x256)
        x256Block = DBlock(inputChannels: io256.i, outputChannels: io256.o,
                           enableSpectralNormalization: enableSN)
        
        let io128 = ioChannels(for: .x128)
        x128Block = DBlock(inputChannels: io128.i, outputChannels: io128.o,
                           enableSpectralNormalization: enableSN)
        
        let io64 = ioChannels(for: .x64)
        x64Block = DBlock(inputChannels: io64.i, outputChannels: io64.o,
                          enableSpectralNormalization: enableSN)
        
        let io32 = ioChannels(for: .x32)
        x32Block = DBlock(inputChannels: io32.i, outputChannels: io32.o,
                          enableSpectralNormalization: enableSN)
        
        let io16 = ioChannels(for: .x16)
        x16Block = DBlock(inputChannels: io16.i, outputChannels: io16.o,
                          enableSpectralNormalization: enableSN)
        
        let io8 = ioChannels(for: .x8)
        x8Block = DBlock(inputChannels: io8.i, outputChannels: io8.o,
                         enableSpectralNormalization: enableSN)
        
        meanConv = SNConv2D(filterShape: (1, 1, io8.o, config.encodedSize),
                            spectralNormalizationEnabled: enableSN,
                            filterInitializer: heNormal())
        logVarConv = SNConv2D(filterShape: (1, 1, io8.o, config.encodedSize),
                              spectralNormalizationEnabled: enableSN,
                              filterInitializer: heNormal())
        
        tail = SNConv2D(filterShape: (4, 4, config.encodedSize, 1),
                        spectralNormalizationEnabled: enableSN,
                      filterInitializer: heNormal())
        
        norm = InstanceNorm(featureCount: io8.o)
    }
    
    struct Output: Differentiable {
        var logit: Tensor<Float>
        var mean: Tensor<Float>
        var logVar: Tensor<Float>
    }
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Output {
        var x = input
        
        x = fromRGB(x)
        
        if imageSize >= .x256 {
            x = x256Block(x)
        }
        if imageSize >= .x128 {
            x = x128Block(x)
        }
        if imageSize >= .x64 {
            x = x64Block(x)
        }
        if imageSize >= .x32 {
            x = x32Block(x)
        }
        if imageSize >= .x16 {
            x = x16Block(x)
        }
        if imageSize >= .x8 {
            x = x8Block(x)
        }
        
        // Added normalization.
        // The variance of the output of resnet can be large.
        // It leads extremely large KL divergence.
        x = norm(leakyRelu(x))
        
        // [batchSize, encodedSize]
        let mean = meanConv(x)
        let logVar = logVarConv(x)
        
        if reparametrize {
            x = Tensor(randomNormal: mean.shape) * exp(0.5 * logVar) + mean
        } else {
            // FIXME:
            x = mean + 0*logVar
        }
        
        // Original impl has leakyRelu here.
//        x = tail(leakyRelu(x)).squeezingShape(at: 1, 2)
        x = tail(x).squeezingShape(at: 1, 2)
        
        return Output(logit: x, mean: mean, logVar: logVar)
    }
}
