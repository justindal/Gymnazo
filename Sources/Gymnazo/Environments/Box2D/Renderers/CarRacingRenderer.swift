#if canImport(SwiftUI) && canImport(CoreGraphics)
import SwiftUI
import CoreGraphics
import MLX

/// SwiftUI view for rendering the CarRacing environment.
public struct CarRacingView: View {
    let snapshot: CarRacingSnapshot
    
    public init(snapshot: CarRacingSnapshot) {
        self.snapshot = snapshot
    }
    
    public var body: some View {
        Canvas { context, size in
            drawBackground(context: context, size: size)
            drawGrass(context: context, size: size)
            drawRoad(context: context, size: size)
            drawCar(context: context, size: size)
            drawIndicators(context: context, size: size)
        }
        .frame(width: 600, height: 400)
    }
    
    private func drawBackground(context: GraphicsContext, size: CGSize) {
        let bgColor = Color(
            red: Double(snapshot.bgColor.0) / 255.0,
            green: Double(snapshot.bgColor.1) / 255.0,
            blue: Double(snapshot.bgColor.2) / 255.0
        )
        context.fill(
            Path(CGRect(origin: .zero, size: size)),
            with: .color(bgColor)
        )
    }
    
    private func drawGrass(context: GraphicsContext, size: CGSize) {
        let grassColor = Color(
            red: Double(snapshot.grassColor.0) / 255.0,
            green: Double(snapshot.grassColor.1) / 255.0,
            blue: Double(snapshot.grassColor.2) / 255.0
        )
        
        let grassDim: CGFloat = CGFloat(TrackConstants.grassDim)
        let angle = CGFloat(-snapshot.carAngle)
        let zoom = CGFloat(snapshot.zoom)
        let transX = CGFloat(snapshot.scrollX)
        let transY = CGFloat(snapshot.scrollY)
        
        for x in stride(from: -20, through: 20, by: 2) {
            for y in stride(from: -20, through: 20, by: 2) {
                let poly = [
                    CGPoint(x: grassDim * CGFloat(x) + grassDim, y: grassDim * CGFloat(y)),
                    CGPoint(x: grassDim * CGFloat(x), y: grassDim * CGFloat(y)),
                    CGPoint(x: grassDim * CGFloat(x), y: grassDim * CGFloat(y) + grassDim),
                    CGPoint(x: grassDim * CGFloat(x) + grassDim, y: grassDim * CGFloat(y) + grassDim)
                ]
                
                let transformed = poly.map { point -> CGPoint in
                    let rotated = rotatePoint(point, by: angle)
                    return CGPoint(
                        x: rotated.x * zoom + transX + size.width / 2,
                        y: size.height - (rotated.y * zoom + transY + size.height / 4)
                    )
                }
                
                var path = Path()
                path.move(to: transformed[0])
                for i in 1..<transformed.count {
                    path.addLine(to: transformed[i])
                }
                path.closeSubpath()
                context.fill(path, with: .color(grassColor))
            }
        }
    }
    
    private func drawRoad(context: GraphicsContext, size: CGSize) {
        let angle = CGFloat(-snapshot.carAngle)
        let zoom = CGFloat(snapshot.zoom)
        let transX = CGFloat(snapshot.scrollX)
        let transY = CGFloat(snapshot.scrollY)
        
        for (i, (polyX, polyY)) in snapshot.roadPoly.enumerated() {
            guard i < snapshot.roadColors.count else { continue }
            let color = snapshot.roadColors[i]
            let drawColor = Color(
                red: Double(min(color.0, 255)) / 255.0,
                green: Double(min(color.1, 255)) / 255.0,
                blue: Double(min(color.2, 255)) / 255.0
            )
            
            guard polyX.count >= 3 && polyY.count >= 3 else { continue }
            
            var points: [CGPoint] = []
            for j in 0..<min(polyX.count, polyY.count) {
                let point = CGPoint(x: CGFloat(polyX[j]), y: CGFloat(polyY[j]))
                let rotated = rotatePoint(point, by: angle)
                let transformed = CGPoint(
                    x: rotated.x * zoom + transX + size.width / 2,
                    y: size.height - (rotated.y * zoom + transY + size.height / 4)
                )
                points.append(transformed)
            }
            
            var path = Path()
            path.move(to: points[0])
            for j in 1..<points.count {
                path.addLine(to: points[j])
            }
            path.closeSubpath()
            context.fill(path, with: .color(drawColor))
        }
    }
    
    private func drawCar(context: GraphicsContext, size: CGSize) {
        let angle = CGFloat(-snapshot.carAngle)
        let zoom = CGFloat(snapshot.zoom)
        let transX = CGFloat(snapshot.scrollX)
        let transY = CGFloat(snapshot.scrollY)
        
        let hullPolys: [[(Float, Float)]] = [
            CarConstants.hullPoly1,
            CarConstants.hullPoly2,
            CarConstants.hullPoly3,
            CarConstants.hullPoly4
        ]
        
        let carWorldX = CGFloat(snapshot.carX)
        let carWorldY = CGFloat(snapshot.carY)
        let carAngle = CGFloat(snapshot.carAngle)
        
        for poly in hullPolys {
            var points: [CGPoint] = []
            for (px, py) in poly {
                let localX = CGFloat(px) * CGFloat(CarConstants.size)
                let localY = CGFloat(py) * CGFloat(CarConstants.size)
                
                let cos_a = cos(carAngle)
                let sin_a = sin(carAngle)
                let worldX = carWorldX + localX * cos_a - localY * sin_a
                let worldY = carWorldY + localX * sin_a + localY * cos_a
                
                let viewPoint = CGPoint(x: worldX, y: worldY)
                let rotated = rotatePoint(viewPoint, by: angle)
                let transformed = CGPoint(
                    x: rotated.x * zoom + transX + size.width / 2,
                    y: size.height - (rotated.y * zoom + transY + size.height / 4)
                )
                points.append(transformed)
            }
            
            guard points.count >= 3 else { continue }
            
            var path = Path()
            path.move(to: points[0])
            for j in 1..<points.count {
                path.addLine(to: points[j])
            }
            path.closeSubpath()
            context.fill(path, with: .color(.red))
        }
        
        for wheel in snapshot.wheelPositions {
            drawWheel(context: context, size: size, wheel: wheel, viewAngle: angle, zoom: zoom, transX: transX, transY: transY)
        }
    }
    
    private func drawWheel(
        context: GraphicsContext,
        size: CGSize,
        wheel: (x: Float, y: Float, angle: Float, phase: Float, omega: Float),
        viewAngle: CGFloat,
        zoom: CGFloat,
        transX: CGFloat,
        transY: CGFloat
    ) {
        let wheelPoly: [(Float, Float)] = [
            (-CarConstants.wheelW, +CarConstants.wheelR),
            (+CarConstants.wheelW, +CarConstants.wheelR),
            (+CarConstants.wheelW, -CarConstants.wheelR),
            (-CarConstants.wheelW, -CarConstants.wheelR)
        ]
        
        let wheelAngle = CGFloat(wheel.angle)
        let wheelX = CGFloat(wheel.x)
        let wheelY = CGFloat(wheel.y)
        
        var points: [CGPoint] = []
        for (px, py) in wheelPoly {
            let localX = CGFloat(px) * CGFloat(CarConstants.size)
            let localY = CGFloat(py) * CGFloat(CarConstants.size)
            
            let cos_a = cos(wheelAngle)
            let sin_a = sin(wheelAngle)
            let worldX = wheelX + localX * cos_a - localY * sin_a
            let worldY = wheelY + localX * sin_a + localY * cos_a
            
            let viewPoint = CGPoint(x: worldX, y: worldY)
            let rotated = rotatePoint(viewPoint, by: viewAngle)
            let transformed = CGPoint(
                x: rotated.x * zoom + transX + size.width / 2,
                y: size.height - (rotated.y * zoom + transY + size.height / 4)
            )
            points.append(transformed)
        }
        
        guard points.count >= 3 else { return }
        
        var path = Path()
        path.move(to: points[0])
        for j in 1..<points.count {
            path.addLine(to: points[j])
        }
        path.closeSubpath()
        context.fill(path, with: .color(.black))
    }
    
    private func drawIndicators(context: GraphicsContext, size: CGSize) {
        let barHeight = size.height * 0.125
        let barRect = CGRect(x: 0, y: size.height - barHeight, width: size.width, height: barHeight)
        context.fill(Path(barRect), with: .color(.black))
        
        let s = size.width / 40.0
        let h = size.height / 40.0
        
        let speedHeight = min(CGFloat(snapshot.trueSpeed) * 0.02 * h, 4 * h)
        let speedRect = CGRect(
            x: 5 * s,
            y: size.height - h - speedHeight,
            width: s,
            height: speedHeight
        )
        context.fill(Path(speedRect), with: .color(.white))
        
        for (i, wheel) in snapshot.wheelPositions.enumerated() {
            let omegaHeight = min(abs(CGFloat(wheel.omega)) * 0.01 * h, 4 * h)
            let omegaRect = CGRect(
                x: CGFloat(7 + i) * s,
                y: size.height - h - omegaHeight,
                width: s,
                height: omegaHeight
            )
            let omegaColor: Color = i < 2 ? .blue : Color(red: 0.2, green: 0, blue: 1)
            context.fill(Path(omegaRect), with: .color(omegaColor))
        }
        
        let steerWidth = CGFloat(snapshot.steeringAngle) * -10.0 * s
        let steerRect = CGRect(
            x: 20 * s,
            y: size.height - 4 * h,
            width: steerWidth,
            height: 2 * h
        )
        context.fill(Path(steerRect), with: .color(.green))
        
        let gyroWidth = CGFloat(snapshot.gyro) * -0.8 * s
        let gyroRect = CGRect(
            x: 30 * s,
            y: size.height - 4 * h,
            width: gyroWidth,
            height: 2 * h
        )
        context.fill(Path(gyroRect), with: .color(.red))
    }
    
    private func rotatePoint(_ point: CGPoint, by angle: CGFloat) -> CGPoint {
        let cos_a = cos(angle)
        let sin_a = sin(angle)
        return CGPoint(
            x: point.x * cos_a - point.y * sin_a,
            y: point.x * sin_a + point.y * cos_a
        )
    }
}

/// Renderer for producing CGImage outputs from CarRacing snapshots.
public struct CarRacingRenderer {
    
    /// Render the snapshot to a CGImage at the specified size.
    @MainActor
    public static func render(snapshot: CarRacingSnapshot, size: CGSize) -> CGImage? {
        guard #available(macOS 13.0, iOS 16.0, *) else {
            return nil
        }
        
        let view = CarRacingView(snapshot: snapshot)
            .frame(width: size.width, height: size.height)
        
        let renderer = ImageRenderer(content: view)
        
        #if os(macOS)
        renderer.scale = 1.0
        #else
        renderer.scale = 1.0
        #endif
        
        return renderer.cgImage
    }
    
    /// Render the snapshot to rgb_array format (600x400).
    @MainActor
    public static func renderRGBArray(snapshot: CarRacingSnapshot) -> CGImage? {
        return render(snapshot: snapshot, size: CGSize(width: 600, height: 400))
    }
    
    /// Render the snapshot to state_pixels format (96x96).
    @MainActor
    public static func renderStatePixels(snapshot: CarRacingSnapshot) -> CGImage? {
        guard let fullImage = render(snapshot: snapshot, size: CGSize(width: 600, height: 400)) else {
            return nil
        }
        return fullImage.scaled(to: CGSize(width: 96, height: 96))
    }
    
    /// Render the snapshot to an MLXArray observation (96x96x3 uint8).
    @MainActor
    public static func renderObservation(snapshot: CarRacingSnapshot) -> MLXArray? {
        guard let stateImage = renderStatePixels(snapshot: snapshot) else {
            return nil
        }
        return stateImage.toMLXArray()
    }
}
#endif

