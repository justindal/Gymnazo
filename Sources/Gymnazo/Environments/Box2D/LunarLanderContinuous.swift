//
//  LunarLanderContinuous.swift
//

import Box2D
import Foundation
import MLX

/// The continuous action variant of the LunarLander environment.
///
/// This variant uses a continuous action space with two dimensions:
/// - Main engine throttle: -1.0 to 0.0 is off, 0.0 to +1.0 throttles from 50% to 100%
/// - Lateral engines: -1.0 to -0.5 fires left, +0.5 to +1.0 fires right
///
/// ## Description
///
/// This environment is a rocket trajectory optimization problem. The goal is to land the rocket
/// on the landing pad smoothly. The landing pad is always at coordinates (0, 0). The rocket
/// starts at the top center of the viewport with a random initial force applied to its center
/// of mass.
///
/// The rocket has three engines: a main engine at the bottom, and two side (lateral) engines.
/// The main engine provides upward thrust, while the side engines provide rotational control.
///
/// ## Action Space
///
/// The action is a continuous vector with shape `(2,)`:
///
/// | Num | Action          | Min  | Max | Unit          |
/// |-----|-----------------|------|-----|---------------|
/// | 0   | Main throttle   | -1.0 | 1.0 | throttle      |
/// | 1   | Lateral control | -1.0 | 1.0 | throttle      |
///
/// - Main engine: Values ≤ 0 turn off the engine. Values > 0 fire at 50% + value × 50% power.
/// - Lateral engines: |value| ≤ 0.5 turns off. |value| > 0.5 fires at proportional power.
///   Positive values fire right engine (rotate left), negative fires left engine (rotate right).
///
/// ## Observation Space
///
/// The observation is an `MLXArray` with shape `(8,)` where the elements correspond to:
///
/// | Num | Observation           | Min    | Max   | Unit     |
/// |-----|-----------------------|--------|-------|----------|
/// | 0   | x position            | -2.5   | 2.5   | position |
/// | 1   | y position            | -2.5   | 2.5   | position |
/// | 2   | x velocity            | -10.0  | 10.0  | velocity |
/// | 3   | y velocity            | -10.0  | 10.0  | velocity |
/// | 4   | angle                 | -2π    | 2π    | radians  |
/// | 5   | angular velocity      | -10.0  | 10.0  | rad/s    |
/// | 6   | left leg contact      | 0      | 1     | bool     |
/// | 7   | right leg contact     | 0      | 1     | bool     |
///
/// ## Rewards
///
/// The reward for moving from the top of the screen to the landing pad with zero velocity
/// is approximately 100-140 points. If the lander crashes, it receives -100. If it lands
/// successfully, it receives +100. Each leg with ground contact gives +10.
///
/// Fuel is infinite, but firing engines applies a penalty:
/// - Main engine: -0.30 per frame while firing
/// - Side engines: -0.03 per frame while firing
///
/// ## Starting State
///
/// The lander starts at the top center of the viewport with a small random velocity.
///
/// ## Episode End
///
/// The episode ends when:
/// 1. **Termination:** The lander crashes (contacts ground with body) or goes out of bounds
/// 2. **Termination:** The lander lands successfully (both legs contact, velocity near zero)
/// 3. **Truncation:** Episode length exceeds the time limit
///
/// ## Arguments
///
/// - `render_mode`: The render mode (`"human"` or `"rgb_array"`).
/// - `gravity`: Gravitational constant (must be between -12.0 and 0.0). Default: `-10.0`
/// - `enableWind`: Whether to enable wind effects. Default: `false`
/// - `windPower`: Maximum wind magnitude. Default: `15.0`
/// - `turbulencePower`: Maximum turbulence magnitude. Default: `1.5`
///
/// ## Usage
///
/// ```swift
/// var env = LunarLanderContinuous()
/// let (obs, info) = env.reset()
/// let action = MLXArray([0.5, 0.0] as [Float32])  // 75% main thrust, no lateral
/// let (nextObs, reward, terminated, truncated, _) = env.step(action)
/// ```
///
/// - Note: For the discrete variant, use ``LunarLander`` directly or create via
///   `make("LunarLander")`.
public struct LunarLanderContinuous: Env {
    public typealias Observation = MLXArray
    public typealias Action = MLXArray
    
    private static let fps: Float = 50.0
    private static let scale: Float = 30.0
    private static let mainEnginePower: Float = 13.0
    private static let sideEnginePower: Float = 0.6
    private static let initialRandom: Float = 1000.0
    private static let legAway: Float = 20
    private static let legDown: Float = 18
    private static let legW: Float = 2
    private static let legH: Float = 8
    private static let legSpringTorque: Float = 40
    private static let sideEngineHeight: Float = 14
    private static let sideEngineAway: Float = 12
    private static let mainEngineYLocation: Float = 4
    private static let viewportW: Float = 600
    private static let viewportH: Float = 400
    private static let chunks: Int = 11
    private static let landerDensity: Float = 5.0
    
    private static let landerCategory: UInt64 = 0x0010
    private static let groundCategory: UInt64 = 0x0001
    private static let legsCategory: UInt64 = 0x0020
    
    public let gravity: Float
    public let enableWind: Bool
    public let windPower: Float
    public let turbulencePower: Float
    
    private var worldId: b2WorldId?
    private var landerId: b2BodyId?
    private var leftLegId: b2BodyId?
    private var rightLegId: b2BodyId?
    private var groundShapeIds: [b2ShapeId] = []
    
    private var terrainX: [Float] = []
    private var terrainY: [Float] = []
    private var helipadX1: Float = 0
    private var helipadX2: Float = 0
    private var helipadY: Float = 0
    private var gameOver: Bool = false
    private var prevShaping: Float?
    private var windIdx: Int = 0
    private var torqueIdx: Int = 0
    private var leftLegContact: Bool = false
    private var rightLegContact: Bool = false
    
    private var landingStableTime: Float = 0
    
    private var lastMainPower: Float = 0
    private var lastSidePower: Float = 0
    
    public let actionSpace: Box
    public let observationSpace: Box
    
    public var spec: EnvSpec? = nil
    public var renderMode: String? = nil
    
    private var _key: MLXArray?
    
    public static var metadata: [String: Any] {
        [
            "render_modes": ["human", "rgb_array"],
            "render_fps": Int(fps),
        ]
    }
    
    public init(
        renderMode: String? = nil,
        gravity: Float = -10.0,
        enableWind: Bool = false,
        windPower: Float = 15.0,
        turbulencePower: Float = 1.5
    ) {
        precondition(gravity > -12.0 && gravity < 0.0,
                     "gravity must be between -12.0 and 0.0, got \(gravity)")
        
        self.renderMode = renderMode
        self.gravity = gravity
        self.enableWind = enableWind
        self.windPower = windPower
        self.turbulencePower = turbulencePower
        
        self.actionSpace = Box(low: -1, high: 1, shape: [2], dtype: .float32)
        
        let low = MLXArray([
            -2.5,   // x position
            -2.5,   // y position
            -10.0,  // x velocity
            -10.0,  // y velocity
            -2 * Float.pi, // angle
            -10.0,  // angular velocity
            0.0,    // left leg contact
            0.0     // right leg contact
        ] as [Float32])
        let high = MLXArray([
            2.5,
            2.5,
            10.0,
            10.0,
            2 * Float.pi,
            10.0,
            1.0,
            1.0
        ] as [Float32])
        self.observationSpace = Box(low: low, high: high, dtype: .float32)
    }
    
    public mutating func step(_ action: Action) -> Step<Observation> {
        guard let worldId = worldId, let landerId = landerId else {
            fatalError("Call reset() before step()")
        }
        
        let mainAction = clip(action[0].item(Float.self), min: -1.0, max: 1.0)
        let lateralAction = clip(action[1].item(Float.self), min: -1.0, max: 1.0)
        
        if enableWind && !leftLegContact && !rightLegContact {
            let windMag = tanh(sin(0.02 * Float(windIdx)) + sin(Float.pi * 0.01 * Float(windIdx))) * windPower
            windIdx += 1
            
            let torqueMag = tanh(sin(0.02 * Float(torqueIdx)) + sin(Float.pi * 0.01 * Float(torqueIdx))) * turbulencePower
            torqueIdx += 1
            
            let windForce = b2Vec2(x: windMag, y: 0)
            b2Body_ApplyForceToCenter(landerId, windForce, true)
            b2Body_ApplyTorque(landerId, torqueMag, true)
        }
        
        let angle = b2Rot_GetAngle(b2Body_GetRotation(landerId))
        let tip = (sin(angle), cos(angle))
        let side = (-tip.1, tip.0)
        
        let (k1, k2, nextKey) = splitKey3(_key!)
        _key = nextKey
        let dispersion0 = MLX.uniform(low: -1.0, high: 1.0, [1], key: k1)[0].item(Float.self) / Self.scale
        let dispersion1 = MLX.uniform(low: -1.0, high: 1.0, [1], key: k2)[0].item(Float.self) / Self.scale
        
        var mPower: Float = 0.0
        if mainAction > 0.0 {
            mPower = (clip(mainAction, min: 0.0, max: 1.0) + 1.0) * 0.5
            
            let ox = tip.0 * (Self.mainEngineYLocation / Self.scale + 2 * dispersion0) + side.0 * dispersion1
            let oy = -tip.1 * (Self.mainEngineYLocation / Self.scale + 2 * dispersion0) - side.1 * dispersion1
            
            let impulseX = -ox * Self.mainEnginePower * mPower
            let impulseY = -oy * Self.mainEnginePower * mPower
            
            let pos = b2Body_GetPosition(landerId)
            let impulsePoint = b2Vec2(x: pos.x, y: pos.y)
            b2Body_ApplyLinearImpulse(landerId, b2Vec2(x: impulseX, y: impulseY), impulsePoint, true)
        }
        lastMainPower = mPower
        
        var sPower: Float = 0.0
        if abs(lateralAction) > 0.5 {
            let direction: Float = lateralAction > 0 ? 1.0 : -1.0
            sPower = clip(abs(lateralAction), min: 0.5, max: 1.0)
            
            let ox = tip.0 * dispersion0 + side.0 * (3 * dispersion1 + direction * Self.sideEngineAway / Self.scale)
            let oy = -tip.1 * dispersion0 - side.1 * (3 * dispersion1 + direction * Self.sideEngineAway / Self.scale)
            
            let impulseX = -ox * Self.sideEnginePower * sPower
            let impulseY = -oy * Self.sideEnginePower * sPower
            
            let pos = b2Body_GetPosition(landerId)
            let impulsePointX = pos.x + ox - tip.0 * 17 / Self.scale
            let impulsePointY = pos.y + oy + tip.1 * Self.sideEngineHeight / Self.scale
            b2Body_ApplyLinearImpulse(landerId, b2Vec2(x: impulseX, y: impulseY), b2Vec2(x: impulsePointX, y: impulsePointY), true)
        }
        lastSidePower = sPower * (lateralAction > 0 ? 1.0 : -1.0)
        
        b2World_Step(worldId, 1.0 / Self.fps, 8)
        
        updateLegContacts()
        checkBodyContact()
        maybeForceSleepForLanding(timeStep: 1.0 / Self.fps)
        
        let obs = getObservation()
        let stateArray = obs.asArray(Float.self)
        
        let shaping = -100.0 * sqrt(stateArray[0] * stateArray[0] + stateArray[1] * stateArray[1])
                    - 100.0 * sqrt(stateArray[2] * stateArray[2] + stateArray[3] * stateArray[3])
                    - 100.0 * abs(stateArray[4])
                    + 10.0 * stateArray[6]
                    + 10.0 * stateArray[7]
        
        var reward: Float = 0.0
        if let prev = prevShaping {
            reward = shaping - prev
        }
        prevShaping = shaping
        
        reward -= mPower * 0.30
        reward -= sPower * 0.03
        
        var terminated = false
        
        if gameOver || abs(stateArray[0]) >= 1.0 {
            terminated = true
            reward = -100
        }
        
        let isAwake = b2Body_IsAwake(landerId)
        if !isAwake {
            terminated = true
            reward = 100
        }
        
        return Step(
            obs: obs,
            reward: Double(reward),
            terminated: terminated,
            truncated: false,
            info: renderMode == nil ? Info() : [
                "lander_awake": .bool(isAwake),
                "left_leg_contact": .bool(leftLegContact),
                "right_leg_contact": .bool(rightLegContact),
                "landing_stable_time": .double(Double(landingStableTime)),
            ]
        )
    }
    
    public mutating func reset(seed: UInt64? = nil, options: [String: Any]? = nil) -> Reset<Observation> {
        if let seed = seed {
            _key = MLX.key(seed)
        } else if _key == nil {
            _key = MLX.key(UInt64.random(in: 0..<UInt64.max))
        }
        
        destroyWorld()
        
        var worldDef = b2DefaultWorldDef()
        worldDef.gravity = b2Vec2(x: 0, y: gravity)
        worldId = b2CreateWorld(&worldDef)
        
        gameOver = false
        prevShaping = nil
        leftLegContact = false
        rightLegContact = false
        landingStableTime = 0
        
        let W = Self.viewportW / Self.scale
        let H = Self.viewportH / Self.scale
        
        let (terrainKey, nextKey) = MLX.split(key: _key!)
        _key = nextKey
        
        var heights = [Float](repeating: 0, count: Self.chunks + 1)
        let heightArray = MLX.uniform(low: 0, high: H / 2, [Self.chunks + 1], key: terrainKey)
        for i in 0...Self.chunks {
            heights[i] = heightArray[i].item(Float.self)
        }
        
        var chunkX = [Float](repeating: 0, count: Self.chunks)
        for i in 0..<Self.chunks {
            chunkX[i] = W / Float(Self.chunks - 1) * Float(i)
        }
        
        helipadX1 = chunkX[Self.chunks / 2 - 1]
        helipadX2 = chunkX[Self.chunks / 2 + 1]
        helipadY = H / 4
        
        heights[Self.chunks / 2 - 2] = helipadY
        heights[Self.chunks / 2 - 1] = helipadY
        heights[Self.chunks / 2] = helipadY
        heights[Self.chunks / 2 + 1] = helipadY
        heights[Self.chunks / 2 + 2] = helipadY
        
        var smoothY = [Float](repeating: 0, count: Self.chunks)
        for i in 0..<Self.chunks {
            let prev = i > 0 ? heights[i - 1] : heights[i]
            let curr = heights[i]
            let next = i < Self.chunks ? heights[i + 1] : heights[i]
            smoothY[i] = 0.33 * (prev + curr + next)
        }
        
        terrainX = chunkX
        terrainY = smoothY
        
        createTerrain()
        createLander()
        
        let (forceKey1, forceKey2, finalKey) = splitKey3(_key!)
        _key = finalKey
        
        let forceX = MLX.uniform(low: -Self.initialRandom, high: Self.initialRandom, [1], key: forceKey1)[0].item(Float.self)
        let forceY = MLX.uniform(low: -Self.initialRandom, high: Self.initialRandom, [1], key: forceKey2)[0].item(Float.self)
        
        if let landerId = landerId {
            b2Body_ApplyForceToCenter(landerId, b2Vec2(x: forceX, y: forceY), true)
        }
        
        if enableWind {
            let (windKey, turbKey, newKey) = splitKey3(_key!)
            _key = newKey
            windIdx = Int(MLX.randInt(low: -9999, high: 9999, key: windKey).item(Int32.self))
            torqueIdx = Int(MLX.randInt(low: -9999, high: 9999, key: turbKey).item(Int32.self))
        }
        
        lastMainPower = 0
        lastSidePower = 0

        b2World_Step(worldId!, 1.0 / Self.fps, 8)
        updateLegContacts()
        checkBodyContact()
        let obs = getObservation()
        prevShaping = computeShaping(from: obs)
        
        return Reset(obs: obs, info: [:])
    }
    
    @discardableResult
    public func render() -> Any? {
        guard let mode = renderMode else { return nil }
        
        switch mode {
        case "human":
            return currentSnapshot
        case "rgb_array":
            return nil
        default:
            return nil
        }
    }
    
    public var unwrapped: any Env { self }
    
    public mutating func close() {
        destroyWorld()
    }
    
    public var currentSnapshot: LunarLanderSnapshot? {
        guard let landerId = landerId,
              let leftLegId = leftLegId,
              let rightLegId = rightLegId else { return nil }
        
        let pos = b2Body_GetPosition(landerId)
        let rot = b2Body_GetRotation(landerId)
        
        let leftLegPos = b2Body_GetPosition(leftLegId)
        let leftLegRot = b2Body_GetRotation(leftLegId)
        let rightLegPos = b2Body_GetPosition(rightLegId)
        let rightLegRot = b2Body_GetRotation(rightLegId)
        
        return LunarLanderSnapshot(
            x: pos.x,
            y: pos.y,
            angle: b2Rot_GetAngle(rot),
            leftLegX: leftLegPos.x,
            leftLegY: leftLegPos.y,
            leftLegAngle: b2Rot_GetAngle(leftLegRot),
            rightLegX: rightLegPos.x,
            rightLegY: rightLegPos.y,
            rightLegAngle: b2Rot_GetAngle(rightLegRot),
            leftLegContact: leftLegContact,
            rightLegContact: rightLegContact,
            mainEnginePower: lastMainPower,
            sideEnginePower: lastSidePower,
            terrainX: terrainX,
            terrainY: terrainY,
            helipadX1: helipadX1,
            helipadX2: helipadX2,
            helipadY: helipadY,
            gameOver: gameOver
        )
    }
    
    private mutating func destroyWorld() {
        if let worldId = worldId, b2World_IsValid(worldId) {
            b2DestroyWorld(worldId)
        }
        self.worldId = nil
        landerId = nil
        leftLegId = nil
        rightLegId = nil
        groundShapeIds = []
    }
    
    private mutating func createTerrain() {
        guard let worldId = worldId else { return }
        
        var groundBodyDef = b2DefaultBodyDef()
        groundBodyDef.type = b2_staticBody
        let groundBodyId = b2CreateBody(worldId, &groundBodyDef)
        
        var shapeDef = b2DefaultShapeDef()
        shapeDef.material.friction = 0.1
        shapeDef.filter.categoryBits = Self.groundCategory
        
        for i in 0..<(terrainX.count - 1) {
            var segment = b2Segment()
            segment.point1 = b2Vec2(x: terrainX[i], y: terrainY[i])
            segment.point2 = b2Vec2(x: terrainX[i + 1], y: terrainY[i + 1])
            let shapeId = b2CreateSegmentShape(groundBodyId, &shapeDef, &segment)
            groundShapeIds.append(shapeId)
        }
    }
    
    private mutating func createLander() {
        guard let worldId = worldId else { return }
        
        let initialX = Self.viewportW / Self.scale / 2
        let initialY = Self.viewportH / Self.scale
        
        var bodyDef = b2DefaultBodyDef()
        bodyDef.type = b2_dynamicBody
        bodyDef.position = b2Vec2(x: initialX, y: initialY)
        landerId = b2CreateBody(worldId, &bodyDef)
        
        var landerVerts: [b2Vec2] = [
            b2Vec2(x: -14 / Self.scale, y: 17 / Self.scale),
            b2Vec2(x: -17 / Self.scale, y: 0 / Self.scale),
            b2Vec2(x: -17 / Self.scale, y: -10 / Self.scale),
            b2Vec2(x: 17 / Self.scale, y: -10 / Self.scale),
            b2Vec2(x: 17 / Self.scale, y: 0 / Self.scale),
            b2Vec2(x: 14 / Self.scale, y: 17 / Self.scale)
        ]
        
        var hull = landerVerts.withUnsafeMutableBufferPointer { ptr in
            b2ComputeHull(ptr.baseAddress, Int32(ptr.count))
        }
        var landerPoly = b2MakePolygon(&hull, 0)
        
        var landerShapeDef = b2DefaultShapeDef()
        landerShapeDef.density = Self.landerDensity
        landerShapeDef.material.friction = 0.1
        landerShapeDef.material.restitution = 0.0
        landerShapeDef.filter.categoryBits = Self.landerCategory
        landerShapeDef.filter.maskBits = Self.groundCategory
        
        _ = b2CreatePolygonShape(landerId!, &landerShapeDef, &landerPoly)
        
        createLegs()
    }
    
    private mutating func createLegs() {
        guard let worldId = worldId, let landerId = landerId else { return }
        
        let landerPos = b2Body_GetPosition(landerId)
        
        for i in [-1, 1] {
            let sign: Float = Float(i)
            
            var legBodyDef = b2DefaultBodyDef()
            legBodyDef.type = b2_dynamicBody
            legBodyDef.position = b2Vec2(
                x: landerPos.x - sign * Self.legAway / Self.scale,
                y: landerPos.y - Self.legDown / Self.scale
            )
            legBodyDef.rotation = b2MakeRot(sign * 0.05)
            
            let legId = b2CreateBody(worldId, &legBodyDef)
            
            var legBox = b2MakeBox(Self.legW / Self.scale, Self.legH / Self.scale)
            
            var legShapeDef = b2DefaultShapeDef()
            legShapeDef.density = 1.0
            legShapeDef.material.restitution = 0.0
            legShapeDef.filter.categoryBits = Self.legsCategory
            legShapeDef.filter.maskBits = Self.groundCategory
            
            _ = b2CreatePolygonShape(legId, &legShapeDef, &legBox)
            
            var jointDef = b2DefaultRevoluteJointDef()
            jointDef.bodyIdA = landerId
            jointDef.bodyIdB = legId
            jointDef.localAnchorA = b2Vec2(x: 0, y: 0)
            jointDef.localAnchorB = b2Vec2(x: sign * Self.legAway / Self.scale, y: Self.legDown / Self.scale)
            jointDef.enableLimit = true

            if i == -1 {
                jointDef.lowerAngle = 0.9 - 0.5  // +0.4
                jointDef.upperAngle = 0.9
            } else {
                jointDef.lowerAngle = -0.9
                jointDef.upperAngle = -0.9 + 0.5  // -0.4
            }
            // Use motor like Gymnasium instead of spring
            jointDef.enableMotor = true
            jointDef.maxMotorTorque = Self.legSpringTorque
            jointDef.motorSpeed = 0.3 * sign
            
            _ = b2CreateRevoluteJoint(worldId, &jointDef)
            
            if i == -1 {
                leftLegId = legId
            } else {
                rightLegId = legId
            }
        }
    }
    
    private mutating func updateLegContacts() {
        guard let leftLegId = leftLegId, let rightLegId = rightLegId else { return }
        
        leftLegContact = checkBodyGroundContact(leftLegId)
        rightLegContact = checkBodyGroundContact(rightLegId)
    }

    private func approximateSleepVelocity(bodyId: b2BodyId) -> Float {
        let v = b2Body_GetLinearVelocity(bodyId)
        let w = b2Body_GetAngularVelocity(bodyId)

        let aabb = b2Body_ComputeAABB(bodyId)
        let width = aabb.upperBound.x - aabb.lowerBound.x
        let height = aabb.upperBound.y - aabb.lowerBound.y
        let maxExtent = 0.5 * sqrt(width * width + height * height)

        let linearSpeed = sqrt(v.x * v.x + v.y * v.y)
        return linearSpeed + abs(w) * maxExtent
    }

    private mutating func maybeForceSleepForLanding(timeStep dt: Float) {
        guard gameOver == false else {
            landingStableTime = 0
            return
        }

        guard leftLegContact || rightLegContact else {
            landingStableTime = 0
            return
        }

        guard let landerId = landerId, let leftLegId = leftLegId, let rightLegId = rightLegId else {
            landingStableTime = 0
            return
        }

        let ids: [b2BodyId] = [landerId, leftLegId, rightLegId]
        let isSleepy = ids.allSatisfy { id in
            let threshold = b2Body_GetSleepThreshold(id)
            return approximateSleepVelocity(bodyId: id) <= threshold
        }

        if isSleepy {
            landingStableTime += dt
        } else {
            landingStableTime = 0
        }

        if landingStableTime >= 0.5 {
            b2Body_SetAwake(landerId, false)
            b2Body_SetAwake(leftLegId, false)
            b2Body_SetAwake(rightLegId, false)
        }
    }
    
    private func checkBodyGroundContact(_ bodyId: b2BodyId) -> Bool {
        let capacity = b2Body_GetContactCapacity(bodyId)
        guard capacity > 0 else { return false }
        
        var contactData = [b2ContactData](repeating: b2ContactData(), count: Int(capacity))
        let count = contactData.withUnsafeMutableBufferPointer { ptr in
            b2Body_GetContactData(bodyId, ptr.baseAddress, capacity)
        }
        
        for i in 0..<Int(count) {
            let contact = contactData[i]
            
            guard contact.manifold.pointCount > 0 else { continue }
            
            let shapeA = contact.shapeIdA
            let shapeB = contact.shapeIdB
            
            let filterA = b2Shape_GetFilter(shapeA)
            let filterB = b2Shape_GetFilter(shapeB)
            
            if filterA.categoryBits == Self.groundCategory || filterB.categoryBits == Self.groundCategory {
                return true
            }
        }
        return false
    }
    
    private mutating func checkBodyContact() {
        guard let landerId = landerId else { return }
        
        let capacity = b2Body_GetContactCapacity(landerId)
        guard capacity > 0 else { return }
        
        var contactData = [b2ContactData](repeating: b2ContactData(), count: Int(capacity))
        let count = contactData.withUnsafeMutableBufferPointer { ptr in
            b2Body_GetContactData(landerId, ptr.baseAddress, capacity)
        }
        
        for i in 0..<Int(count) {
            let contact = contactData[i]
            
            guard contact.manifold.pointCount > 0 else { continue }
            
            let shapeA = contact.shapeIdA
            let shapeB = contact.shapeIdB
            
            let filterA = b2Shape_GetFilter(shapeA)
            let filterB = b2Shape_GetFilter(shapeB)
            
            if filterA.categoryBits == Self.groundCategory || filterB.categoryBits == Self.groundCategory {
                gameOver = true
                return
            }
        }
    }
    
    private func getObservation() -> MLXArray {
        guard let landerId = landerId else {
            fatalError("State is nil - call reset() first")
        }
        
        let pos = b2Body_GetPosition(landerId)
        let vel = b2Body_GetLinearVelocity(landerId)
        let rot = b2Body_GetRotation(landerId)
        let angVel = b2Body_GetAngularVelocity(landerId)
        
        let W = Self.viewportW / Self.scale
        let H = Self.viewportH / Self.scale
        
        let normX = (pos.x - W / 2) / (W / 2)
        let normY = (pos.y - (helipadY + Self.legDown / Self.scale)) / (H / 2)
        
        let normVx = vel.x * (W / 2) / Self.fps
        let normVy = vel.y * (H / 2) / Self.fps
        
        let angle = b2Rot_GetAngle(rot)
        let normAngVel = 20.0 * angVel / Self.fps
        
        return MLXArray([
            normX,
            normY,
            normVx,
            normVy,
            angle,
            normAngVel,
            leftLegContact ? 1.0 : 0.0,
            rightLegContact ? 1.0 : 0.0
        ] as [Float32])
    }
    
    private func clip(_ value: Float, min minVal: Float, max maxVal: Float) -> Float {
        Swift.min(Swift.max(value, minVal), maxVal)
    }
    
    private func splitKey3(_ key: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (k1, rest) = MLX.split(key: key)
        let (k2, k3) = MLX.split(key: rest)
        return (k1, k2, k3)
    }
    
    private func computeShaping(from obs: MLXArray) -> Float {
        let s = obs.asArray(Float.self)
        return -100.0 * sqrt(s[0] * s[0] + s[1] * s[1])
             - 100.0 * sqrt(s[2] * s[2] + s[3] * s[3])
             - 100.0 * abs(s[4])
             + 10.0 * s[6]
             + 10.0 * s[7]
    }
}
