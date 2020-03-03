import Foundation
import TensorFlow
import GANUtils
import ImageLoader
import TensorBoardX

Context.local.randomSeed = (42, 42)
let rng = XorshiftRandomNumberGenerator()

let imageSize: ImageSize = .x256
let latentSize = 256
let batchSize = 16

let config = Config(
    loss: .nonSaturating,
    batchSize: batchSize,
    learningRates: GDPair(G: 1e-4, D: 1e-4),
    alpha: 1e-5,
    Ic: 0.1,
    reparameterizeInGTraining: false,
    imageSize: imageSize,
    G: Generator.Config(
        latentSize: latentSize,
        resizeMethod: .bilinear,
        enableBatchNorm: false
    ),
    D: Discriminator.Config(
        encodedSize: 256,
        enableSpectralNormalization: true
    )
)

var generator = Generator(config: config.G, imageSize: imageSize)
var discriminator = Discriminator(config: config.D, imageSize: imageSize)

let optG = RMSProp(for: generator, learningRate: config.learningRates.D, rho: 0.99)
let optD = RMSProp(for: discriminator, learningRate: config.learningRates.D, rho: 0.99)
let criterion = GANLoss(config.loss)


// MARK: - Dataset
let args = ProcessInfo.processInfo.arguments
guard args.count == 2 else {
    print("Image directory is not specified.")
    exit(1)
}
print("Search images...")
let imageDir = URL(fileURLWithPath: args[1])
let entries = [Entry](directory: imageDir)
print("\(entries.count) images found")
let loader = ImageLoader(
    entries: entries,
    transforms: [
        Transforms.paddingToSquare(with: 1),
        Transforms.resize(.area, width: imageSize.rawValue, height: imageSize.rawValue),
        Transforms.randomFlipHorizontally()
    ],
    rng: rng
)

// MARK: - Plot
let logdir = URL(fileURLWithPath: "./logdir")
let writer = SummaryWriter(logdir: logdir)
try writer.addJSONText(tag: "config", encodable: config)

// MARK: - Training
func train() {
    Context.local.learningPhase = .training
    
    let inferStep = 10000
    
    var step = 0
    
    for epoch in 0..<1_000_000 {
        loader.shuffle()
        
        for batch in loader.iterator(batchSize: config.batchSize) {
            if step % 1 == 0 {
                print("epoch: \(epoch), step:\(step)")
            }
            
            let reals = 2 * batch.images - 1
            
            trainSingleStep(reals: reals, step: step)
            
            if step % inferStep == 0 {
                infer(step: step)
            }
            
            step += 1
        }
    }
    
    // last inference
    infer(step: step)
}

var beta: Float = 0.1
func trainSingleStep(reals: Tensor<Float>, step: Int) {
    let noise = sampleNoise(size: batchSize, latentSize: latentSize)
    
    let fakePlotPeriod = 1000
    
    // Update discriminator
    discriminator.reparametrize = true
    let 𝛁discriminator = gradient(at: discriminator) { discriminator -> Tensor<Float> in
        let fakes = generator(noise)
        let realOutout = discriminator(reals)
        let fakeOutput = discriminator(fakes)
        
        let ganLoss = criterion.lossD(real: realOutout.logit, fake: fakeOutput.logit)
        let mean = Tensor(concatenating: [realOutout.mean, fakeOutput.mean], alongAxis: 0)
            .reshaped(to: [-1, config.D.encodedSize])
        let logVar = Tensor(concatenating: [realOutout.logVar, fakeOutput.logVar], alongAxis: 0)
            .reshaped(to: [-1, config.D.encodedSize])
        var kld = (mean.squared() + exp(logVar) - logVar - 1)
        kld = kld.sum(squeezingAxes: 1) / 2
        let bottleneckLoss = kld.mean() - config.Ic
        
        let loss = ganLoss + beta * bottleneckLoss
        
        // Update beta
        beta = withoutDerivative(at: max(0, beta + config.alpha * bottleneckLoss.scalarized()))
        
        writer.addScalar(tag: "loss/D", scalar: loss.scalarized(), globalStep: step)
        writer.addScalar(tag: "loss/Dgan", scalar: ganLoss.scalarized(), globalStep: step)
        writer.addScalar(tag: "loss/Dbottleneck", scalar: bottleneckLoss.scalarized(), globalStep: step)
        writer.addScalar(tag: "loss/Dbeta", scalar: beta, globalStep: step)
        
        if step % fakePlotPeriod == 0 {
            writer.plotImages(tag: "reals", images: reals, globalStep: step)
            writer.plotImages(tag: "fakes", images: fakes, globalStep: step)
            writer.flush()
        }
        
        return loss
    }
    optD.update(&discriminator, along: 𝛁discriminator)
    
    // Update Generator
    discriminator.reparametrize = config.reparameterizeInGTraining
    let 𝛁generator = gradient(at: generator) { generator ->Tensor<Float> in
        let fakes = generator(noise)
        let output = discriminator(fakes)
        
        let loss = criterion.lossG(output.logit)
        
        writer.addScalar(tag: "loss/G", scalar: loss.scalarized(), globalStep: step)
        
        return loss
    }
    optG.update(&generator, along: 𝛁generator)
    
}

let truncationFactor: Float = 0.9
let testNoises = (0..<8).map { _ in sampleNoise(size: 64, latentSize: latentSize) * truncationFactor }
let testGridNoises = (0..<8).map { _ in
    makeGrid(corners: sampleNoise(size: 4, latentSize: latentSize) * truncationFactor,
             gridSize: 8, flatten: true)
}

func infer(step: Int) {
    print("infer...")
    
    for (i, noise) in testNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_random/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    for (i, noise) in testGridNoises.enumerated() {
        let reals = generator.inferring(from: noise, batchSize: batchSize)
        writer.plotImages(tag: "test_intpl/\(i)", images: reals, colSize: 8, globalStep: step)
    }
    
    writer.flush()
}

train()
writer.close()
