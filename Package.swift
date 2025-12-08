// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Gymnazo",
    platforms: [
        .macOS(.v14),
        .iOS(.v17)
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Gymnazo",
            targets: ["Gymnazo"]
        ),
        .library(name: "Box2D", targets: ["Box2D"])
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
        .target(name: "Box2D",
            path: "Vendor/Box2D",
            exclude: ["src/box2d.natvis", "src/CMakeLists.txt"],
            sources: ["src"],
            publicHeadersPath: "include",
            cSettings: [
                .headerSearchPath("include")
            ]
        ),
        .target(
            name: "Gymnazo",
            dependencies: [.product(name: "MLX", package: "mlx-swift"),
                           .product(name: "MLXNN", package: "mlx-swift"),
                           .product(name: "MLXOptimizers", package: "mlx-swift"),
                           .product(name: "MLXRandom", package: "mlx-swift"),
                           .product(name: "Collections", package: "swift-collections"),
                            "Box2D"
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
