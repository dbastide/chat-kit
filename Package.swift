// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "ChatKit",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "ChatKit", targets: ["ChatKit"]),
    ],
    targets: [
        .target(name: "ChatKit"),
        .testTarget(name: "ChatKitTests", dependencies: ["ChatKit"]),
    ]
)
