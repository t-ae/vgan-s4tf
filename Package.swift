// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VGAN",
    platforms: [
        .macOS(.v10_13),
    ],
    dependencies: [
        .package(name: "GANUtils", url: "https://github.com/t-ae/gan-utils-s4tf.git", from: "0.3.6"),
        .package(name: "ImageLoader", url: "https://github.com/t-ae/image-loader.git", from: "0.1.9"),
        .package(name: "TensorBoardX", url: "https://github.com/t-ae/tensorboardx-s4tf.git", from: "0.1.2"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages which this package depends on.
        .target(
            name: "VGAN",
            dependencies: ["GANUtils", "ImageLoader", "TensorBoardX"]),
        .testTarget(
            name: "VGANTests",
            dependencies: ["VGAN"]),
    ]
)
