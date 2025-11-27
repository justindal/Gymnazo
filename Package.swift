// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Gymnazo",
    platforms: [
        .macOS(.v14),
        .iOS(.v16)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Gymnazo",
            targets: ["Gymnazo"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.29.1"),
        .package(
            url: "https://github.com/apple/swift-collections.git",
            .upToNextMinor(from: "1.3.0")
        ),
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.0.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "Gymnazo",
            dependencies: [.product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXNN", package: "mlx-swift"),
                           .product(name: "MLXOptimizers", package: "mlx-swift"),
                           .product(name: "MLXRandom", package: "mlx-swift"),
                           .product(name: "Collections", package: "swift-collections")
            ],
            resources: [
                .process("Resources")
            ]
        ),
        .testTarget(
            name: "GymnazoTests",
            dependencies: ["Gymnazo"]
        ),
    ]
)
