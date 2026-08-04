import Foundation

struct FlappyBirdModel: Sendable {
    static let worldWidth: Float = 1
    static let worldHeight: Float = 1
    static let birdX: Float = 0.25
    static let birdWidth: Float = 0.06
    static let birdHeight: Float = 0.05
    static let pipeWidth: Float = 0.12
    static let gapHeight: Float = 0.3
    static let pipeSpacing: Float = 0.55
    static let initialPipeX: Float = 0.9
    static let timeStep: Float = 1 / 30
    static let gravity: Float = -1.8
    static let flapVelocity: Float = 0.65
    static let terminalVelocity: Float = -1

    private(set) var birdY: Float = 0.5
    private(set) var birdVelocity: Float = 0
    private(set) var pipes: [Pipe] = []
    private(set) var score = 0

    private var generator: SeededGenerator?
    private var needsReset = true

    mutating func reset(seed: UInt64?) -> [Float] {
        if let seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0...UInt64.max))
        }

        birdY = 0.5
        birdVelocity = 0
        score = 0
        pipes = (0..<3).map { index in
            Pipe(
                x: Self.initialPipeX + Float(index) * Self.pipeSpacing,
                gapY: nextGapY()
            )
        }
        needsReset = false
        return observation
    }

    mutating func step(flaps: Bool) throws -> Result {
        guard !needsReset else {
            throw GymnazoError.stepBeforeReset
        }

        if flaps {
            birdVelocity = Self.flapVelocity
        }
        birdVelocity = max(
            Self.terminalVelocity,
            birdVelocity + Self.gravity * Self.timeStep
        )
        birdY += birdVelocity * Self.timeStep

        let distance = Self.pipeSpeed * Self.timeStep
        for index in pipes.indices {
            pipes[index].x -= distance
        }

        var passed = 0
        for index in pipes.indices where !pipes[index].passed {
            if pipes[index].x + Self.pipeWidth / 2 < Self.birdX - Self.birdWidth / 2 {
                pipes[index].passed = true
                passed += 1
            }
        }
        score += passed

        recyclePipes()

        let terminated = collided
        needsReset = terminated
        let reward = terminated ? -1 : 0.1 + Double(passed)

        return Result(
            observation: observation,
            reward: reward,
            terminated: terminated,
            score: score
        )
    }

    var snapshot: FlappyBirdSnapshot {
        FlappyBirdSnapshot(
            birdY: min(max(birdY, 0), Self.worldHeight),
            birdVelocity: birdVelocity,
            pipes: pipes.map { FlappyBirdPipeSnapshot(x: $0.x, gapY: $0.gapY) },
            score: score
        )
    }

    private static let pipeSpeed: Float = 0.4

    private var observation: [Float] {
        let pipe = nextPipe
        let velocityScale = max(abs(Self.terminalVelocity), Self.flapVelocity)
        return [
            min(max(birdY / Self.worldHeight * 2 - 1, -1), 1),
            min(max(birdVelocity / velocityScale, -1), 1),
            min(max((pipe.x - Self.birdX) / Self.worldWidth, -1), 1),
            min(max(pipe.gapY / Self.worldHeight * 2 - 1, -1), 1),
        ]
    }

    private var nextPipe: Pipe {
        pipes
            .filter { $0.x + Self.pipeWidth / 2 >= Self.birdX - Self.birdWidth / 2 }
            .min { $0.x < $1.x }
            ?? pipes.min { $0.x < $1.x }
            ?? Pipe(x: Self.initialPipeX, gapY: 0.5)
    }

    private var collided: Bool {
        let birdLeft = Self.birdX - Self.birdWidth / 2
        let birdRight = Self.birdX + Self.birdWidth / 2
        let birdBottom = birdY - Self.birdHeight / 2
        let birdTop = birdY + Self.birdHeight / 2

        if birdBottom <= 0 || birdTop >= Self.worldHeight {
            return true
        }

        return pipes.contains { pipe in
            let pipeLeft = pipe.x - Self.pipeWidth / 2
            let pipeRight = pipe.x + Self.pipeWidth / 2
            guard birdRight >= pipeLeft, birdLeft <= pipeRight else { return false }
            let gapBottom = pipe.gapY - Self.gapHeight / 2
            let gapTop = pipe.gapY + Self.gapHeight / 2
            return birdBottom <= gapBottom || birdTop >= gapTop
        }
    }

    private mutating func recyclePipes() {
        for index in pipes.indices where pipes[index].x + Self.pipeWidth / 2 < 0 {
            let rightmostX = pipes.map(\.x).max() ?? Self.initialPipeX
            pipes[index] = Pipe(x: rightmostX + Self.pipeSpacing, gapY: nextGapY())
        }
    }

    private mutating func nextGapY() -> Float {
        guard var generator else { return 0.5 }
        let margin = Self.gapHeight / 2 + 0.08
        let value = generator.uniform(in: margin...(Self.worldHeight - margin))
        self.generator = generator
        return value
    }

    struct Result: Sendable {
        let observation: [Float]
        let reward: Double
        let terminated: Bool
        let score: Int
    }

    struct Pipe: Sendable {
        var x: Float
        var gapY: Float
        var passed = false
    }
}
