import Box2D
import Foundation

/// Constants for the car physics simulation matching Gymnasium's car_dynamics.py.
public enum CarConstants {
    public static let size: Float = 0.02
    public static let enginePower: Float = 100_000_000 * size * size
    public static let wheelMomentOfInertia: Float = 4000 * size * size
    public static let frictionLimit: Float = 1_000_000 * size * size
    public static let wheelR: Float = 27
    public static let wheelW: Float = 14
    
    public static let wheelPositions: [(Float, Float)] = [
        (-55, +80), (+55, +80), (-55, -82), (+55, -82)
    ]
    
    public static let hullPoly1: [(Float, Float)] = [
        (-60, +130), (+60, +130), (+60, +110), (-60, +110)
    ]
    public static let hullPoly2: [(Float, Float)] = [
        (-15, +120), (+15, +120), (+20, +20), (-20, 20)
    ]
    public static let hullPoly3: [(Float, Float)] = [
        (+25, +20), (+50, -10), (+50, -40), (+20, -90),
        (-20, -90), (-50, -40), (-50, -10), (-25, +20)
    ]
    public static let hullPoly4: [(Float, Float)] = [
        (-50, -120), (+50, -120), (+50, -90), (-50, -90)
    ]
    
    public static let wheelColor: (UInt8, UInt8, UInt8) = (0, 0, 0)
    public static let wheelWhite: (UInt8, UInt8, UInt8) = (77, 77, 77)
    public static let mudColor: (UInt8, UInt8, UInt8) = (102, 102, 0)
    
    public static let wheelCategory: UInt64 = 0x0020
    public static let wheelMask: UInt64 = 0x001
}

/// State for a single wheel of the car.
public struct WheelState {
    public var bodyId: b2BodyId
    public var jointId: b2JointId
    public var wheelRadius: Float
    public var gas: Float = 0
    public var brake: Float = 0
    public var steer: Float = 0
    public var phase: Float = 0
    public var omega: Float = 0
    public var skidStart: b2Vec2?
    public var tiles: Set<Int> = []
    
    public init(bodyId: b2BodyId, jointId: b2JointId, wheelRadius: Float) {
        self.bodyId = bodyId
        self.jointId = jointId
        self.wheelRadius = wheelRadius
    }
}

/// A skid particle for visual trail effects.
public struct SkidParticle {
    public var color: (UInt8, UInt8, UInt8)
    public var ttl: Float
    public var poly: [(Float, Float)]
    public var grass: Bool
    
    public init(color: (UInt8, UInt8, UInt8), ttl: Float, poly: [(Float, Float)], grass: Bool) {
        self.color = color
        self.ttl = ttl
        self.poly = poly
        self.grass = grass
    }
}

/// The car physics model matching Gymnasium's Car class from car_dynamics.py.
public struct Car {
    public private(set) var worldId: b2WorldId
    public private(set) var hullId: b2BodyId
    public var wheels: [WheelState]
    public private(set) var fuelSpent: Float = 0
    public private(set) var particles: [SkidParticle] = []
    
    public var hullColor: (Float, Float, Float) = (0.8, 0.0, 0.0)
    
    public init(worldId: b2WorldId, initAngle: Float, initX: Float, initY: Float) {
        self.worldId = worldId
        self.wheels = []
        
        var hullBodyDef = b2DefaultBodyDef()
        hullBodyDef.type = b2_dynamicBody
        hullBodyDef.position = b2Vec2(x: initX, y: initY)
        hullBodyDef.rotation = b2MakeRot(initAngle)
        self.hullId = b2CreateBody(worldId, &hullBodyDef)
        
        Self.createHullFixtures(hullId: hullId)
        createWheels(initAngle: initAngle, initX: initX, initY: initY)
    }
    
    private static func createHullFixtures(hullId: b2BodyId) {
        let polys: [[(Float, Float)]] = [
            CarConstants.hullPoly1,
            CarConstants.hullPoly2,
            CarConstants.hullPoly3,
            CarConstants.hullPoly4
        ]
        
        for poly in polys {
            var verts = poly.map { b2Vec2(x: $0.0 * CarConstants.size, y: $0.1 * CarConstants.size) }
            var hull = verts.withUnsafeMutableBufferPointer { ptr in
                b2ComputeHull(ptr.baseAddress, Int32(ptr.count))
            }
            var polygon = b2MakePolygon(&hull, 0)
            
            var shapeDef = b2DefaultShapeDef()
            shapeDef.density = 1.0
            _ = b2CreatePolygonShape(hullId, &shapeDef, &polygon)
        }
    }
    
    private mutating func createWheels(initAngle: Float, initX: Float, initY: Float) {
        let wheelPoly: [(Float, Float)] = [
            (-CarConstants.wheelW, +CarConstants.wheelR),
            (+CarConstants.wheelW, +CarConstants.wheelR),
            (+CarConstants.wheelW, -CarConstants.wheelR),
            (-CarConstants.wheelW, -CarConstants.wheelR)
        ]
        
        for (wx, wy) in CarConstants.wheelPositions {
            let frontK: Float = 1.0
            
            var wheelBodyDef = b2DefaultBodyDef()
            wheelBodyDef.type = b2_dynamicBody
            wheelBodyDef.position = b2Vec2(
                x: initX + wx * CarConstants.size,
                y: initY + wy * CarConstants.size
            )
            wheelBodyDef.rotation = b2MakeRot(initAngle)
            let wheelId = b2CreateBody(worldId, &wheelBodyDef)
            
            var verts = wheelPoly.map {
                b2Vec2(x: $0.0 * frontK * CarConstants.size, y: $0.1 * frontK * CarConstants.size)
            }
            var hull = verts.withUnsafeMutableBufferPointer { ptr in
                b2ComputeHull(ptr.baseAddress, Int32(ptr.count))
            }
            var polygon = b2MakePolygon(&hull, 0)
            
            var shapeDef = b2DefaultShapeDef()
            shapeDef.density = 0.1
            shapeDef.filter.categoryBits = CarConstants.wheelCategory
            shapeDef.filter.maskBits = CarConstants.wheelMask
            shapeDef.material.restitution = 0.0
            _ = b2CreatePolygonShape(wheelId, &shapeDef, &polygon)
            
            var jointDef = b2DefaultRevoluteJointDef()
            jointDef.bodyIdA = hullId
            jointDef.bodyIdB = wheelId
            jointDef.localAnchorA = b2Vec2(x: wx * CarConstants.size, y: wy * CarConstants.size)
            jointDef.localAnchorB = b2Vec2(x: 0, y: 0)
            jointDef.enableMotor = true
            jointDef.enableLimit = true
            jointDef.maxMotorTorque = 180 * 900 * CarConstants.size * CarConstants.size
            jointDef.motorSpeed = 0
            jointDef.lowerAngle = -0.4
            jointDef.upperAngle = +0.4
            let jointId = b2CreateRevoluteJoint(worldId, &jointDef)
            
            let wheelRadius = frontK * CarConstants.wheelR * CarConstants.size
            wheels.append(WheelState(bodyId: wheelId, jointId: jointId, wheelRadius: wheelRadius))
        }
    }
    
    /// Apply gas to rear wheels (indices 2 and 3).
    public mutating func gas(_ gasInput: Float) {
        let clampedGas = min(max(gasInput, 0), 1)
        for i in 2...3 {
            let diff = min(clampedGas - wheels[i].gas, 0.1)
            if diff > 0 {
                wheels[i].gas += diff
            } else {
                wheels[i].gas = clampedGas
            }
        }
    }
    
    /// Apply brakes to all wheels.
    public mutating func brake(_ brakeInput: Float) {
        for i in 0..<wheels.count {
            wheels[i].brake = brakeInput
        }
    }
    
    /// Set steering target for front wheels (indices 0 and 1).
    public mutating func steer(_ steerInput: Float) {
        wheels[0].steer = steerInput
        wheels[1].steer = steerInput
    }
    
    /// Perform one physics step for the car dynamics.
    public mutating func step(dt: Float, tileFrictions: [Int: Float]) {
        for i in 0..<wheels.count {
            let jointAngle = b2RevoluteJoint_GetAngle(wheels[i].jointId)
            let dir = (wheels[i].steer - jointAngle) >= 0 ? Float(1) : Float(-1)
            let val = abs(wheels[i].steer - jointAngle)
            b2RevoluteJoint_SetMotorSpeed(wheels[i].jointId, dir * min(50.0 * val, 3.0))
            
            var grass = true
            var frictionLimit = CarConstants.frictionLimit * 0.6
            for tileIdx in wheels[i].tiles {
                if let tileFriction = tileFrictions[tileIdx] {
                    frictionLimit = max(frictionLimit, CarConstants.frictionLimit * tileFriction)
                    grass = false
                }
            }
            
            let wheelBodyId = wheels[i].bodyId
            let forw = b2Body_GetWorldVector(wheelBodyId, b2Vec2(x: 0, y: 1))
            let side = b2Body_GetWorldVector(wheelBodyId, b2Vec2(x: 1, y: 0))
            let v = b2Body_GetLinearVelocity(wheelBodyId)
            
            let vf = forw.x * v.x + forw.y * v.y
            let vs = side.x * v.x + side.y * v.y
            
            wheels[i].omega += dt * CarConstants.enginePower * wheels[i].gas / CarConstants.wheelMomentOfInertia / (abs(wheels[i].omega) + 5.0)
            fuelSpent += dt * CarConstants.enginePower * wheels[i].gas
            
            if wheels[i].brake >= 0.9 {
                wheels[i].omega = 0
            } else if wheels[i].brake > 0 {
                let brakeForce: Float = 15
                let brakeDir: Float = wheels[i].omega >= 0 ? -1 : 1
                var brakeVal = brakeForce * wheels[i].brake
                if abs(brakeVal) > abs(wheels[i].omega) {
                    brakeVal = abs(wheels[i].omega)
                }
                wheels[i].omega += brakeDir * brakeVal
            }
            
            wheels[i].phase += wheels[i].omega * dt
            
            let vr = wheels[i].omega * wheels[i].wheelRadius
            var fForce = (-vf + vr) * 205000 * CarConstants.size * CarConstants.size
            var pForce = -vs * 205000 * CarConstants.size * CarConstants.size
            
            var force = sqrt(fForce * fForce + pForce * pForce)
            
            if abs(force) > 2.0 * frictionLimit {
                handleSkid(wheelIndex: i, grass: grass)
            } else {
                wheels[i].skidStart = nil
            }
            
            if abs(force) > frictionLimit {
                fForce = fForce / force * frictionLimit
                pForce = pForce / force * frictionLimit
                force = frictionLimit
            }
            
            wheels[i].omega -= dt * fForce * wheels[i].wheelRadius / CarConstants.wheelMomentOfInertia
            
            let forceVec = b2Vec2(
                x: pForce * side.x + fForce * forw.x,
                y: pForce * side.y + fForce * forw.y
            )
            let wheelPos = b2Body_GetPosition(wheelBodyId)
            b2Body_ApplyForce(hullId, forceVec, wheelPos, true)
        }
    }
    
    private mutating func handleSkid(wheelIndex: Int, grass: Bool) {
        let pos = b2Body_GetPosition(wheels[wheelIndex].bodyId)
        let currentPos = (pos.x, pos.y)
        
        if wheels[wheelIndex].skidStart == nil {
            wheels[wheelIndex].skidStart = pos
        } else if let start = wheels[wheelIndex].skidStart {
            let color = grass ? CarConstants.mudColor : CarConstants.wheelColor
            let particle = SkidParticle(
                color: color,
                ttl: 1,
                poly: [(start.x, start.y), currentPos],
                grass: grass
            )
            particles.append(particle)
            wheels[wheelIndex].skidStart = nil
            
            while particles.count > 30 {
                particles.removeFirst()
            }
        }
    }
    
    /// Destroy all car bodies from the world.
    public mutating func destroy() {
        if b2Body_IsValid(hullId) {
            b2DestroyBody(hullId)
        }
        for wheel in wheels {
            if b2Body_IsValid(wheel.bodyId) {
                b2DestroyBody(wheel.bodyId)
            }
        }
        wheels.removeAll()
    }
}

