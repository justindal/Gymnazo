// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "Gymnazo",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .library(
            name: "Gymnazo",
            targets: ["Gymnazo"]
        ),
        .library(name: "Box2D", targets: ["Box2D"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            from: "0.29.1"
        ),
        .package(
            url: "https://github.com/apple/swift-collections.git",
            .upToNextMinor(from: "1.3.0")
        ),
        .package(
            url: "https://github.com/apple/swift-docc-plugin",
            from: "1.0.0"
        ),
    ],
    targets: [
        .target(
            name: "Box2D",
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

            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Collections", package: "swift-collections"),
                "Box2D",
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
