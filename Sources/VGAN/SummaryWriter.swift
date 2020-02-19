import Foundation
import TensorFlow
import Python
import TensorBoardX

extension SummaryWriter {
    func plotImages(tag: String,
                    images: Tensor<Float>,
                    colSize: Int = 8,
                    globalStep: Int) {
        var images = images
        images = (images + 1) / 2
        images = images.clipped(min: 0, max: 1)
        addImages(tag: tag,
                  images: images,
                  colSize: colSize,
                  globalStep: globalStep)
    }
}
