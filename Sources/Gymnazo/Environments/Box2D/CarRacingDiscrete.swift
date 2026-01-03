import Box2D
import CoreGraphics
import Foundation
import MLX

/// The discrete CarRacing environment matching Gymnasium's CarRacing-v3 with discrete actions.
///
/// A top-down racing environment where the agent controls a car using discrete
/// actions for steering, gas, and brake. The observation is a 96x96 RGB image.
public struct CarRacingDiscrete: Env {
    public typealias Observation = MLXArray
    public typealias Action = Int
    
    public let action_space: Discrete
    public let observation_space: Box
    
    public var spec: EnvSpec? = nil
    public var render_mode: String? = nil
    
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
    
    private var _key: MLXArray?
    
    nonisolated(unsafe) private static let fallbackObservation: MLXArray = {
        let pixels = [UInt8](repeating: 128, count: 96 * 96 * 3)
        let arr = MLXArray(pixels).reshaped([96, 96, 3]).asType(.uint8)
        eval(arr)
        return arr
    }()
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array", "state_pixels"],
            "render_fps": Int(TrackConstants.fps)
        ]
    }
    
    public init(
        render_mode: String? = nil,
        lapCompletePercent: Float = 0.95,
        domainRandomize: Bool = false
    ) {
        self.render_mode = render_mode
        self.lapCompletePercent = lapCompletePercent
        self.domainRandomize = domainRandomize
        
        self.action_space = Discrete(n: 5)
        
        self.observation_space = Box(
            low: 0,
            high: 255,
            shape: [96, 96, 3],
            dtype: .uint8
        )
        
        initColors()
    }
    
    private mutating func initColors() {
        if domainRandomize {
            let (colorKey, nextKey) = MLX.split(key: _key ?? MLX.key(0))
            _key = nextKey
            let (idxKey, finalKey) = MLX.split(key: nextKey)
            _key = finalKey
            
            let colorArray = MLX.uniform(low: 0, high: 210, [6], key: colorKey)
            let idxArray = MLX.randInt(low: 0, high: 3, key: idxKey)
            eval(colorArray, idxArray)
            
            let colors = colorArray.asArray(Float.self)
            roadColor = (colors[0], colors[1], colors[2])
            bgColor = (colors[3], colors[4], colors[5])
            grassColor = bgColor
            
            let idx = Int(idxArray.item(Int32.self))
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
        
        let (colorKey, nextKey) = MLX.split(key: _key ?? MLX.key(0))
        _key = nextKey
        let (idxKey, finalKey) = MLX.split(key: nextKey)
        _key = finalKey
        
        let colorArray = MLX.uniform(low: 0, high: 210, [6], key: colorKey)
        let idxArray = MLX.randInt(low: 0, high: 3, key: idxKey)
        eval(colorArray, idxArray)
        
        let colors = colorArray.asArray(Float.self)
        roadColor = (colors[0], colors[1], colors[2])
        bgColor = (colors[3], colors[4], colors[5])
        grassColor = bgColor
        
        let idx = Int(idxArray.item(Int32.self))
        switch idx {
        case 0: grassColor.0 += 20
        case 1: grassColor.1 += 20
        default: grassColor.2 += 20
        }
    }
    
    public mutating func step(_ action: Action) -> Step<Observation> {
        guard var car = car, let worldId = worldId, let trackData = trackData else {
            fatalError("Call reset() before step()")
        }
        
        precondition(action_space.contains(action), "Invalid action: \(action)")
        
        let steerValue: Float = -0.6 * (action == 1 ? 1 : 0) + 0.6 * (action == 2 ? 1 : 0)
        let gasValue: Float = 0.2 * (action == 3 ? 1 : 0)
        let brakeValue: Float = 0.8 * (action == 4 ? 1 : 0)
        
        car.steer(steerValue)
        car.gas(gasValue)
        car.brake(brakeValue)
        
        var tileFrictions: [Int: Float] = [:]
        for tile in trackData.tiles {
            tileFrictions[tile.idx] = tile.roadFriction
        }
        
        car.step(dt: 1.0 / TrackConstants.fps, tileFrictions: tileFrictions)
        self.car = car
        
        b2World_Step(worldId, 1.0 / TrackConstants.fps, 8)
        
        t += 1.0 / TrackConstants.fps
        
        updateWheelTileContacts()
        
        let state = renderStatePixels()
        
        var stepReward: Float = 0
        var terminated = false
        var info = Info()
        
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
        
        if render_mode == "human" {
            _ = render()
        }
        
        return Step(
            obs: state,
            reward: Double(stepReward),
            terminated: terminated,
            truncated: false,
            info: info
        )
    }
    
    private mutating func updateWheelTileContacts() {
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
                    if bodyId.index1 == wheelBodyId.index1 && 
                       bodyId.world0 == wheelBodyId.world0 && 
                       bodyId.generation == wheelBodyId.generation {
                        
                        car.wheels[wheelIndex].tiles.insert(k)
                        
                        if !trackData.tiles[k].roadVisited {
                            trackData.tiles[k].roadVisited = true
                            reward += 1000.0 / Float(trackData.tiles.count)
                            tileVisitedCount += 1
                            
                            if trackData.tiles[k].idx == 0 {
                                let visitRatio = Float(tileVisitedCount) / Float(trackData.tiles.count)
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
    
    public mutating func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> Reset<Observation> {
        if let seed = seed {
            _key = MLX.key(seed)
        } else if _key == nil {
            _key = MLX.key(UInt64.random(in: 0..<UInt64.max))
        }
        
        destroy()
        
        if domainRandomize {
            let shouldRandomize = (options?["randomize"] as? Bool) ?? true
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
            trackData = TrackGenerator.createTrack(key: &_key!, roadColor: roadColor)
            if trackData != nil {
                break
            }
            attempts += 1
            if attempts > 100 {
                fatalError("Failed to generate track after 100 attempts")
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
        
        let state = renderStatePixels()
        
        if render_mode == "human" {
            _ = render()
        }
        
        return Reset(obs: state, info: [:])
    }
    
    private final class UnsafeBox<T>: @unchecked Sendable {
        var value: T?
    }
    
    private func renderStatePixels() -> MLXArray {
        #if canImport(SwiftUI)
        if let snapshot = currentSnapshot {
            let box = UnsafeBox<MLXArray>()
            if Thread.isMainThread {
                MainActor.assumeIsolated {
                    box.value = CarRacingRenderer.renderObservation(snapshot: snapshot)
                }
            } else {
                DispatchQueue.main.sync {
                    box.value = CarRacingRenderer.renderObservation(snapshot: snapshot)
                }
            }
            if let obs = box.value {
                return obs
            }
        }
        #endif
        return Self.fallbackObservation
    }
    
    @discardableResult
    public func render() -> Any? {
        guard let mode = render_mode else { return nil }
        
        switch mode {
        case "human":
            return currentSnapshot
        case "rgb_array":
            #if canImport(SwiftUI)
            if let snapshot = currentSnapshot {
                let box = UnsafeBox<CGImage>()
                if Thread.isMainThread {
                    MainActor.assumeIsolated {
                        box.value = CarRacingRenderer.renderRGBArray(snapshot: snapshot)
                    }
                } else {
                    DispatchQueue.main.sync {
                        box.value = CarRacingRenderer.renderRGBArray(snapshot: snapshot)
                    }
                }
                return box.value
            }
            #endif
            return nil
        case "state_pixels":
            #if canImport(SwiftUI)
            if let snapshot = currentSnapshot {
                let box = UnsafeBox<CGImage>()
                if Thread.isMainThread {
                    MainActor.assumeIsolated {
                        box.value = CarRacingRenderer.renderStatePixels(snapshot: snapshot)
                    }
                } else {
                    DispatchQueue.main.sync {
                        box.value = CarRacingRenderer.renderStatePixels(snapshot: snapshot)
                    }
                }
                return box.value
            }
            #endif
            return nil
        default:
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
        
        let zoom = 0.1 * TrackConstants.scale * max(1 - t, 0) + TrackConstants.zoom * TrackConstants.scale * min(t, 1)
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

