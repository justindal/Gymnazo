import Box2D
import CoreGraphics
import Foundation
import MLX

/// Snapshot of the CarRacing environment state for rendering.
public struct CarRacingSnapshot: Equatable, Sendable {
    public let carX: Float
    public let carY: Float
    public let carAngle: Float
    public let wheelPositions: [(x: Float, y: Float, angle: Float, phase: Float, omega: Float)]
    public let roadPoly: [([Float], [Float])]
    public let roadColors: [(Float, Float, Float)]
    public let bgColor: (Float, Float, Float)
    public let grassColor: (Float, Float, Float)
    public let trueSpeed: Float
    public let steeringAngle: Float
    public let gyro: Float
    public let zoom: Float
    public let scrollX: Float
    public let scrollY: Float
    public let t: Float

    public static func == (lhs: CarRacingSnapshot, rhs: CarRacingSnapshot) -> Bool {
        lhs.carX == rhs.carX && lhs.carY == rhs.carY && lhs.carAngle == rhs.carAngle
    }
}

struct CarRacingObservation: Sendable {
    let pixels: [UInt8]
}

/// The continuous CarRacing environment matching Gymnasium's CarRacing-v3.
///
/// A top-down racing environment where the agent controls a car using continuous
/// steering, gas, and brake inputs. The observation is a 96x96 RGB image.
public struct CarRacing: Env {
    public let actionSpace: any Space
    public let observationSpace: any Space

    public var spec: EnvSpec? = nil
    public var renderMode: RenderMode? = nil

    public let lapCompletePercent: Float
    public let domainRandomize: Bool

    private var worldId: b2WorldId?
    private var car: Car?
    private var trackData: TrackData?

    private var roadColor: (Float, Float, Float) = (102, 102, 102)
    private var bgColor: (Float, Float, Float) = (102, 204, 102)
    private var grassColor: (Float, Float, Float) = (102, 230, 102)

    private var reward: Float = 0
    private var prevReward: Float = 0
    private var tileVisitedCount: Int = 0
    private var t: Float = 0
    private var newLap: Bool = false

    private var generator: SeededGenerator?

    private func fallbackObservationBytes() -> [UInt8] {
        [UInt8](repeating: 128, count: 96 * 96 * 3)
    }

    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array", "state_pixels"],
            "render_fps": Int(TrackConstants.fps),
        ]
    }

    public init(
        renderMode: RenderMode? = nil,
        lapCompletePercent: Float = 0.95,
        domainRandomize: Bool = false
    ) {
        self.renderMode = renderMode
        self.lapCompletePercent = lapCompletePercent
        self.domainRandomize = domainRandomize

        self.actionSpace = Box(
            low: MLXArray([-1.0, 0.0, 0.0] as [Float32]),
            high: MLXArray([1.0, 1.0, 1.0] as [Float32]),
            dtype: .float32
        )

        self.observationSpace = Box(
            low: 0,
            high: 255,
            shape: [96, 96, 3],
            dtype: .uint8
        )

        initColors()
    }

    private mutating func initColors() {
        if domainRandomize, var generator {
            let colors = (0..<6).map { _ in generator.uniform(in: 0...210) }
            roadColor = (colors[0], colors[1], colors[2])
            bgColor = (colors[3], colors[4], colors[5])
            grassColor = bgColor

            let idx = Int.random(in: 0..<3, using: &generator)
            self.generator = generator
            switch idx {
            case 0: grassColor.0 += 20
            case 1: grassColor.1 += 20
            default: grassColor.2 += 20
            }
        } else {
            roadColor = (102, 102, 102)
            bgColor = (102, 204, 102)
            grassColor = (102, 230, 102)
        }
    }

    private mutating func reinitColors(randomize: Bool) {
        guard domainRandomize else { return }
        guard randomize else { return }
        initColors()
    }

    public mutating func step(_ action: MLXArray) throws -> Step {
        let result = try stepValues(action.asArray(Float.self))
        let observation = MLXArray(result.observation.pixels).reshaped([96, 96, 3]).asType(.uint8)
        eval(observation)
        return Step(
            obs: observation,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: Info(valueInfo: result.info)
        )
    }

    mutating func stepValues(
        _ action: [Float],
        includeObservation: Bool = true
    ) throws -> ValueStep<CarRacingObservation> {
        guard var car = car, let worldId = worldId, let trackData = trackData else {
            throw GymnazoError.stepBeforeReset
        }
        guard action.count == 3 else {
            throw GymnazoError.invalidAction("CarRacing expects three action values.")
        }
        car.steer(-action[0])
        car.gas(action[1])
        car.brake(action[2])

        var tileFrictions: [Int: Float] = [:]
        for tile in trackData.tiles {
            tileFrictions[tile.idx] = tile.roadFriction
        }

        car.step(dt: 1.0 / TrackConstants.fps, tileFrictions: tileFrictions)
        self.car = car

        b2World_Step(worldId, 1.0 / TrackConstants.fps, 8)

        t += 1.0 / TrackConstants.fps

        let vel = b2Body_GetLinearVelocity(car.hullId)
        let trueSpeed = sqrt(vel.x * vel.x + vel.y * vel.y)
        updateWheelTileContacts(awardReward: trueSpeed > 0.1)

        let state = CarRacingObservation(
            pixels: includeObservation ? renderStateBytes() : []
        )

        var stepReward: Float = 0
        var terminated = false
        var info: ValueInfo = [:]

        reward -= 0.1

        stepReward = reward - prevReward
        prevReward = reward

        if tileVisitedCount == trackData.tiles.count || newLap {
            terminated = true
            info["lap_finished"] = .bool(true)
        }

        let pos = b2Body_GetPosition(car.hullId)
        if abs(pos.x) > TrackConstants.playfield || abs(pos.y) > TrackConstants.playfield {
            terminated = true
            stepReward = -100
            info["lap_finished"] = .bool(false)
        }

        if renderMode == .human {
            _ = try render()
        }

        return ValueStep(
            observation: state,
            reward: Double(stepReward),
            terminated: terminated,
            truncated: false,
            info: info
        )
    }

    private mutating func updateWheelTileContacts(awardReward: Bool) {
        guard var car = car, let trackData = trackData else { return }

        for i in 0..<car.wheels.count {
            car.wheels[i].tiles.removeAll()
        }

        var wheelBodyToIndex: [Int32: Int] = [:]
        for i in 0..<car.wheels.count {
            wheelBodyToIndex[car.wheels[i].bodyId.index1] = i
        }

        for k in 0..<trackData.tiles.count {
            guard let tileShape = trackData.tiles[k].shapeId else { continue }

            let capacity = b2Shape_GetSensorCapacity(tileShape)
            guard capacity > 0 else { continue }

            var overlaps = [b2ShapeId](repeating: b2ShapeId(), count: Int(capacity))
            let overlapCount = overlaps.withUnsafeMutableBufferPointer { ptr in
                b2Shape_GetSensorOverlaps(tileShape, ptr.baseAddress, capacity)
            }

            for j in 0..<Int(overlapCount) {
                let overlappingShape = overlaps[j]
                guard b2Shape_IsValid(overlappingShape) else { continue }

                let bodyId = b2Shape_GetBody(overlappingShape)

                if let wheelIndex = wheelBodyToIndex[bodyId.index1] {
                    let wheelBodyId = car.wheels[wheelIndex].bodyId
                    if bodyId.index1 == wheelBodyId.index1 && bodyId.world0 == wheelBodyId.world0
                        && bodyId.generation == wheelBodyId.generation
                    {

                        car.wheels[wheelIndex].tiles.insert(k)

                        if !trackData.tiles[k].roadVisited {
                            trackData.tiles[k].roadVisited = true
                            tileVisitedCount += 1
                            if awardReward {
                                reward += 1000.0 / Float(trackData.tiles.count)
                            }

                            if trackData.tiles[k].idx == 0 {
                                let visitRatio =
                                    Float(tileVisitedCount) / Float(trackData.tiles.count)
                                if visitRatio > lapCompletePercent {
                                    newLap = true
                                }
                            }
                        }
                    }
                }
            }
        }

        self.car = car
    }

    public mutating func reset(seed: UInt64? = nil, options: EnvOptions? = nil) throws -> Reset {
        let randomize = options?["randomize"] as? Bool
        let result = try resetValues(seed: seed, randomize: randomize)
        let observation = MLXArray(result.observation.pixels).reshaped([96, 96, 3]).asType(.uint8)
        eval(observation)
        return Reset(obs: observation, info: Info(valueInfo: result.info))
    }

    mutating func resetValues(
        seed: UInt64? = nil,
        randomize: Bool? = nil,
        includeObservation: Bool = true
    ) throws -> ValueReset<CarRacingObservation> {
        if let seed = seed {
            generator = SeededGenerator(seed: seed)
        } else if generator == nil {
            generator = SeededGenerator(seed: UInt64.random(in: 0..<UInt64.max))
        }

        destroy()

        if domainRandomize {
            let shouldRandomize = randomize ?? true
            reinitColors(randomize: shouldRandomize)
        }

        var worldDef = b2DefaultWorldDef()
        worldDef.gravity = b2Vec2(x: 0, y: 0)
        worldId = b2CreateWorld(&worldDef)

        reward = 0
        prevReward = 0
        tileVisitedCount = 0
        t = 0
        newLap = false

        var attempts = 0
        while true {
            guard var generator else {
                throw GymnazoError.invalidState("Missing random generator")
            }
            trackData = TrackGenerator.createTrack(
                generator: &generator,
                roadColor: roadColor
            )
            self.generator = generator
            if trackData != nil {
                break
            }
            attempts += 1
            if attempts > 100 {
                throw GymnazoError.operationFailed("Failed to generate track after 100 attempts")
            }
        }

        TrackGenerator.createTileBodies(worldId: worldId!, tiles: &trackData!.tiles)

        let startTrack = trackData!.track[0]
        car = Car(
            worldId: worldId!,
            initAngle: startTrack.beta,
            initX: startTrack.x,
            initY: startTrack.y
        )

        let state = CarRacingObservation(
            pixels: includeObservation ? renderStateBytes() : []
        )

        if renderMode == .human {
            _ = try render()
        }

        return ValueReset(observation: state)
    }

    private func renderStatePixels() -> MLXArray {
        let observation = MLXArray(renderStateBytes()).reshaped([96, 96, 3]).asType(.uint8)
        eval(observation)
        return observation
    }

    private func renderStateBytes() -> [UInt8] {
        #if canImport(SwiftUI)
            guard Thread.isMainThread else { return fallbackObservationBytes() }
            if let snapshot = currentSnapshot {
                let pixels = MainActor.assumeIsolated {
                    CarRacingRenderer.renderObservationBytes(snapshot: snapshot)
                }
                if let pixels { return pixels }
            }
        #endif
        return fallbackObservationBytes()
    }

    @discardableResult
    public func render() throws -> RenderOutput? {
        guard let mode = renderMode else { return nil }

        switch mode {
        case .human:
            if let snapshot = currentSnapshot {
                return .other(snapshot)
            }
            return nil
        case .rgbArray:
            #if canImport(SwiftUI)
                guard Thread.isMainThread else { return nil }
                if let snapshot = currentSnapshot {
                    let image = MainActor.assumeIsolated {
                        CarRacingRenderer.renderRGBArray(snapshot: snapshot)
                    }
                    if let image { return .rgbArray(image) }
                }
            #endif
            return nil
        case .statePixels:
            #if canImport(SwiftUI)
                guard Thread.isMainThread else { return nil }
                if let snapshot = currentSnapshot {
                    let image = MainActor.assumeIsolated {
                        CarRacingRenderer.renderStatePixels(snapshot: snapshot)
                    }
                    if let image { return .statePixels(image) }
                }
            #endif
            return nil
        case .ansi:
            return nil
        }
    }

    public var currentSnapshot: CarRacingSnapshot? {
        guard let car = car else { return nil }

        let hullPos = b2Body_GetPosition(car.hullId)
        let hullRot = b2Body_GetRotation(car.hullId)
        let hullAngle = b2Rot_GetAngle(hullRot)

        var wheelPositions: [(x: Float, y: Float, angle: Float, phase: Float, omega: Float)] = []
        for wheel in car.wheels {
            let pos = b2Body_GetPosition(wheel.bodyId)
            let rot = b2Body_GetRotation(wheel.bodyId)
            wheelPositions.append((pos.x, pos.y, b2Rot_GetAngle(rot), wheel.phase, wheel.omega))
        }

        var roadPolyX: [[Float]] = []
        var roadPolyY: [[Float]] = []
        var roadColors: [(Float, Float, Float)] = []

        if let trackData = trackData {
            for (vertices, color) in trackData.roadPoly {
                roadPolyX.append(vertices.map { $0.0 })
                roadPolyY.append(vertices.map { $0.1 })
                roadColors.append(color)
            }
        }

        let vel = b2Body_GetLinearVelocity(car.hullId)
        let trueSpeed = sqrt(vel.x * vel.x + vel.y * vel.y)
        let steeringAngle = b2RevoluteJoint_GetAngle(car.wheels[0].jointId)
        let gyro = b2Body_GetAngularVelocity(car.hullId)

        let zoom =
            0.1 * TrackConstants.scale * max(1 - t, 0) + TrackConstants.zoom * TrackConstants.scale
            * min(t, 1)
        let cameraAngle = -hullAngle
        let rawScrollX = -hullPos.x * zoom
        let rawScrollY = -hullPos.y * zoom
        let scrollX = rawScrollX * cos(cameraAngle) - rawScrollY * sin(cameraAngle)
        let scrollY = rawScrollX * sin(cameraAngle) + rawScrollY * cos(cameraAngle)

        return CarRacingSnapshot(
            carX: hullPos.x,
            carY: hullPos.y,
            carAngle: hullAngle,
            wheelPositions: wheelPositions,
            roadPoly: zip(roadPolyX, roadPolyY).map { ($0, $1) },
            roadColors: roadColors,
            bgColor: bgColor,
            grassColor: grassColor,
            trueSpeed: trueSpeed,
            steeringAngle: steeringAngle,
            gyro: gyro,
            zoom: zoom,
            scrollX: scrollX,
            scrollY: scrollY,
            t: t
        )
    }

    public var unwrapped: any Env { self }

    public mutating func close() {
        destroy()
    }

    private mutating func destroy() {
        car?.destroy()
        car = nil

        if let worldId = worldId, b2World_IsValid(worldId) {
            b2DestroyWorld(worldId)
        }
        self.worldId = nil
        trackData = nil
    }
}
