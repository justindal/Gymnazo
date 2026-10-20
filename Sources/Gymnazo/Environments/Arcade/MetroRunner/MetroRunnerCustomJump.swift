struct MetroRunnerCustomJump: Sendable {
    let curve: MetroRunnerCurve
    let startY: Double
    let startZ: Double
    let height: Double
    let speed: Double
    let duration: Double
    var elapsed = 0.0
    let source: MetroRunnerEntityKind
    let disallowJump: Bool
    let fastDive: Bool
}
