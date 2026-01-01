import Box2D
import Foundation
import MLX

/// Constants for the CarRacing track matching Gymnasium's car_racing.py.
public enum TrackConstants {
    public static let scale: Float = 6.0
    public static let trackRad: Float = 900 / scale
    public static let playfield: Float = 2000 / scale
    public static let fps: Float = 50
    public static let zoom: Float = 2.7
    
    public static let trackDetailStep: Float = 21 / scale
    public static let trackTurnRate: Float = 0.31
    public static let trackWidth: Float = 40 / scale
    public static let border: Float = 8 / scale
    public static let borderMinCount: Int = 4
    public static let grassDim: Float = playfield / 20.0
    
    public static let checkpoints: Int = 12
    
    public static let tileCategory: UInt64 = 0x0001
}

/// A road tile on the track.
public final class RoadTile {
    public let idx: Int
    public var roadFriction: Float
    public var roadVisited: Bool
    public var color: (Float, Float, Float)
    public let vertices: [(Float, Float)]
    public var shapeId: b2ShapeId?
    
    public init(idx: Int, roadFriction: Float, color: (Float, Float, Float), vertices: [(Float, Float)]) {
        self.idx = idx
        self.roadFriction = roadFriction
        self.roadVisited = false
        self.color = color
        self.vertices = vertices
    }
}

/// Track generation result containing all road data.
public struct TrackData {
    public var track: [(alpha: Float, beta: Float, x: Float, y: Float)]
    public var tiles: [RoadTile]
    public var roadPoly: [(vertices: [(Float, Float)], color: (Float, Float, Float))]
    public var startAlpha: Float
    
    public init() {
        self.track = []
        self.tiles = []
        self.roadPoly = []
        self.startAlpha = 0
    }
}

/// Track generator matching Gymnasium's _create_track() logic.
public struct TrackGenerator {
    
    /// Generate a random track using the provided MLX key for randomness.
    public static func createTrack(
        key: inout MLXArray,
        roadColor: (Float, Float, Float),
        verbose: Bool = false
    ) -> TrackData? {
        var data = TrackData()
        
        let (checkpointKey, nextKey) = MLX.split(key: key)
        key = nextKey
        
        var checkpoints: [(Float, Float, Float)] = []
        let noiseArray = MLX.uniform(low: 0, high: Float(2 * Float.pi / Float(TrackConstants.checkpoints)), [TrackConstants.checkpoints], key: checkpointKey)
        let (radKey, nextKey2) = MLX.split(key: key)
        key = nextKey2
        let radArray = MLX.uniform(low: TrackConstants.trackRad / 3, high: TrackConstants.trackRad, [TrackConstants.checkpoints], key: radKey)
        
        eval(noiseArray, radArray)
        let noiseValues = noiseArray.asArray(Float.self)
        let radValues = radArray.asArray(Float.self)
        
        for c in 0..<TrackConstants.checkpoints {
            var alpha = 2 * Float.pi * Float(c) / Float(TrackConstants.checkpoints) + noiseValues[c]
            var rad = radValues[c]
            
            if c == 0 {
                alpha = 0
                rad = 1.5 * TrackConstants.trackRad
            }
            if c == TrackConstants.checkpoints - 1 {
                alpha = 2 * Float.pi * Float(c) / Float(TrackConstants.checkpoints)
                data.startAlpha = 2 * Float.pi * (-0.5) / Float(TrackConstants.checkpoints)
                rad = 1.5 * TrackConstants.trackRad
            }
            
            checkpoints.append((alpha, rad * cos(alpha), rad * sin(alpha)))
        }
        
        var x: Float = 1.5 * TrackConstants.trackRad
        var y: Float = 0
        var beta: Float = 0
        var destI = 0
        var laps = 0
        var track: [(Float, Float, Float, Float)] = []
        var noFreeze = 2500
        var visitedOtherSide = false
        
        while true {
            var alpha = atan2(y, x)
            if visitedOtherSide && alpha > 0 {
                laps += 1
                visitedOtherSide = false
            }
            if alpha < 0 {
                visitedOtherSide = true
                alpha += 2 * Float.pi
            }
            
            while true {
                var failed = true
                while true {
                    let (destAlpha, _, _) = checkpoints[destI % checkpoints.count]
                    if alpha <= destAlpha {
                        failed = false
                        break
                    }
                    destI += 1
                    if destI % checkpoints.count == 0 {
                        break
                    }
                }
                if !failed {
                    break
                }
                alpha -= 2 * Float.pi
            }
            
            let (_, destX, destY) = checkpoints[destI % checkpoints.count]
            
            let r1x = cos(beta)
            let r1y = sin(beta)
            let p1x = -r1y
            let p1y = r1x
            
            let destDx = destX - x
            let destDy = destY - y
            var proj = r1x * destDx + r1y * destDy
            
            while beta - alpha > 1.5 * Float.pi {
                beta -= 2 * Float.pi
            }
            while beta - alpha < -1.5 * Float.pi {
                beta += 2 * Float.pi
            }
            
            let prevBeta = beta
            proj *= TrackConstants.scale
            
            if proj > 0.3 {
                beta -= min(TrackConstants.trackTurnRate, abs(0.001 * proj))
            }
            if proj < -0.3 {
                beta += min(TrackConstants.trackTurnRate, abs(0.001 * proj))
            }
            
            x += p1x * TrackConstants.trackDetailStep
            y += p1y * TrackConstants.trackDetailStep
            track.append((alpha, prevBeta * 0.5 + beta * 0.5, x, y))
            
            if laps > 4 {
                break
            }
            noFreeze -= 1
            if noFreeze == 0 {
                break
            }
        }
        
        var i1 = -1
        var i2 = -1
        var i = track.count
        
        while true {
            i -= 1
            if i == 0 {
                return nil
            }
            
            let passThroughStart = track[i].0 > data.startAlpha && track[i - 1].0 <= data.startAlpha
            if passThroughStart && i2 == -1 {
                i2 = i
            } else if passThroughStart && i1 == -1 {
                i1 = i
                break
            }
        }
        
        guard i1 != -1 && i2 != -1 else {
            return nil
        }
        
        if verbose {
            print("Track generation: \(i1)..\(i2) -> \(i2 - i1)-tiles track")
        }
        
        let slicedTrack = Array(track[i1..<(i2 - 1)])
        
        let firstBeta = slicedTrack[0].1
        let firstPerpX = cos(firstBeta)
        let firstPerpY = sin(firstBeta)
        let dx = slicedTrack[0].2 - slicedTrack[slicedTrack.count - 1].2
        let dy = slicedTrack[0].3 - slicedTrack[slicedTrack.count - 1].3
        let wellGluedTogether = sqrt(pow(firstPerpX * dx, 2) + pow(firstPerpY * dy, 2))
        
        if wellGluedTogether > TrackConstants.trackDetailStep {
            return nil
        }
        
        var border = [Bool](repeating: false, count: slicedTrack.count)
        
        for i in 0..<slicedTrack.count {
            var good = true
            var oneside = 0
            for neg in 0..<TrackConstants.borderMinCount {
                let idx1 = (i - neg + slicedTrack.count) % slicedTrack.count
                let idx2 = (i - neg - 1 + slicedTrack.count) % slicedTrack.count
                let beta1 = slicedTrack[idx1].1
                let beta2 = slicedTrack[idx2].1
                good = good && abs(beta1 - beta2) > TrackConstants.trackTurnRate * 0.2
                oneside += beta1 - beta2 >= 0 ? 1 : -1
            }
            good = good && abs(oneside) == TrackConstants.borderMinCount
            border[i] = good
        }
        
        for i in 0..<slicedTrack.count {
            for neg in 0..<TrackConstants.borderMinCount {
                let idx = (i - neg + slicedTrack.count) % slicedTrack.count
                border[idx] = border[idx] || border[i]
            }
        }
        
        for i in 0..<slicedTrack.count {
            let (_, beta1, x1, y1) = slicedTrack[i]
            let prevIdx = (i - 1 + slicedTrack.count) % slicedTrack.count
            let (_, beta2, x2, y2) = slicedTrack[prevIdx]
            
            let road1L = (x1 - TrackConstants.trackWidth * cos(beta1), y1 - TrackConstants.trackWidth * sin(beta1))
            let road1R = (x1 + TrackConstants.trackWidth * cos(beta1), y1 + TrackConstants.trackWidth * sin(beta1))
            let road2L = (x2 - TrackConstants.trackWidth * cos(beta2), y2 - TrackConstants.trackWidth * sin(beta2))
            let road2R = (x2 + TrackConstants.trackWidth * cos(beta2), y2 + TrackConstants.trackWidth * sin(beta2))
            
            let vertices = [road1L, road1R, road2R, road2L]
            
            let c = 0.01 * Float(i % 3) * 255
            let tileColor = (roadColor.0 + c, roadColor.1 + c, roadColor.2 + c)
            
            let tile = RoadTile(idx: i, roadFriction: 1.0, color: tileColor, vertices: vertices)
            data.tiles.append(tile)
            data.roadPoly.append((vertices: vertices, color: tileColor))
            
            if border[i] {
                let side: Float = beta2 - beta1 >= 0 ? 1 : -1
                let b1L = (
                    x1 + side * TrackConstants.trackWidth * cos(beta1),
                    y1 + side * TrackConstants.trackWidth * sin(beta1)
                )
                let b1R = (
                    x1 + side * (TrackConstants.trackWidth + TrackConstants.border) * cos(beta1),
                    y1 + side * (TrackConstants.trackWidth + TrackConstants.border) * sin(beta1)
                )
                let b2L = (
                    x2 + side * TrackConstants.trackWidth * cos(beta2),
                    y2 + side * TrackConstants.trackWidth * sin(beta2)
                )
                let b2R = (
                    x2 + side * (TrackConstants.trackWidth + TrackConstants.border) * cos(beta2),
                    y2 + side * (TrackConstants.trackWidth + TrackConstants.border) * sin(beta2)
                )
                
                let borderColor: (Float, Float, Float) = i % 2 == 0 ? (255, 255, 255) : (255, 0, 0)
                data.roadPoly.append((vertices: [b1L, b1R, b2R, b2L], color: borderColor))
            }
        }
        
        data.track = slicedTrack.map { ($0.0, $0.1, $0.2, $0.3) }
        return data
    }
    
    /// Create Box2D sensor shapes for all tiles and attach userData.
    public static func createTileBodies(
        worldId: b2WorldId,
        tiles: inout [RoadTile]
    ) {
        var groundBodyDef = b2DefaultBodyDef()
        groundBodyDef.type = b2_staticBody
        let groundBodyId = b2CreateBody(worldId, &groundBodyDef)
        
        for i in 0..<tiles.count {
            let tile = tiles[i]
            var verts = tile.vertices.map { b2Vec2(x: $0.0, y: $0.1) }
            
            var hull = verts.withUnsafeMutableBufferPointer { ptr in
                b2ComputeHull(ptr.baseAddress, Int32(ptr.count))
            }
            
            guard hull.count >= 3 else { continue }
            
            var polygon = b2MakePolygon(&hull, 0)
            
            var shapeDef = b2DefaultShapeDef()
            shapeDef.isSensor = true
            shapeDef.filter.categoryBits = TrackConstants.tileCategory
            
            let shapeId = b2CreatePolygonShape(groundBodyId, &shapeDef, &polygon)
            tiles[i].shapeId = shapeId
        }
    }
}

