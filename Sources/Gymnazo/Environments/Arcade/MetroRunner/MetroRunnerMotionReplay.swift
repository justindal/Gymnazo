import Foundation

/// Replays a captured clear-track scenario through the native Swift simulator.
///
/// This fixture requires a centered, grounded, unpowered start at track height zero without contacts.
/// It retains the capture's initial position and time; it does not establish original-game fidelity.
public enum MetroRunnerMotionReplay {
    public static let targetUUID = "4f9aeb54-055f-3708-89f0-6405a08627f9"

    /// Writes a simulator JSONL trace without overwriting an existing file.
    public static func replay(original: URL, output: URL, flatTrack: Bool) throws {
        guard flatTrack else { throw ReplayError("Explicit flat-track confirmation is required.") }
        let data = try Data(contentsOf: original)
        let decoder = JSONDecoder()
        var headers: [Header] = [], samples: [Sample] = [], actions: [Action] = []
        for line in data.split(separator: 10) where !line.allSatisfy({ $0 == 13 || $0 == 32 || $0 == 9 }) {
            let row = Data(line)
            switch try decoder.decode(Kind.self, from: row).kind {
            case "header": headers.append(try decoder.decode(Header.self, from: row))
            case "sample": samples.append(try decoder.decode(Sample.self, from: row))
            case "action": actions.append(try decoder.decode(Action.self, from: row))
            case "error": throw ReplayError("The capture contains an error.")
            default: break
            }
        }
        guard headers.count == 1, let header = headers.first, header.schemaVersion == 1,
            header.source == "original_game", header.targetUUID.lowercased() == targetUUID,
            !header.scenario.isEmpty, samples.count >= 3 else {
            throw ReplayError("Expected one original-game schema-1 header for the supported target and at least three samples.")
        }
        for sample in samples {
            guard [sample.t, sample.x, sample.y, sample.z, sample.vy, sample.vz].allSatisfy(\.isFinite), sample.t >= 0 else {
                throw ReplayError("The capture contains invalid motion values.")
            }
        }
        for (a, b) in zip(samples, samples.dropFirst()) {
            guard b.t > a.t, b.t - a.t <= 0.1, abs(b.z - a.z) <= 100 else {
                throw ReplayError("The capture contains a time gap, discontinuity, or reset.")
            }
        }
        let first = samples[0]
        guard first.grounded, !samples.contains(where: \.flying), abs(first.vy) <= 0.01,
            abs(first.x) <= 20.05, abs(first.y) <= 0.05 else {
            throw ReplayError("Start grounded at track height zero and centered in a lane, without flight or vertical motion.")
        }
        let lane = Int((first.x / 20).rounded()) + 1
        guard (0..<3).contains(lane), abs(Double(lane - 1) * 20 - first.x) <= 0.05 else {
            throw ReplayError("The initial player position is not centered in a lane.")
        }
        let duration = samples[samples.count - 1].t - first.t
        guard duration <= 300 else { throw ReplayError("Capture a motion scenario of at most 300 seconds.") }
        let commands: [String: MetroRunnerAction] = ["left": .left, "right": .right, "jump": .jump, "down": .down]
        var byTick: [Int: Action] = [:]
        for action in actions {
            guard action.t.isFinite, commands[action.action] != nil else { throw ReplayError("Unsupported capture action.") }
            let offset = (action.t - first.t) * 60
            guard offset.isFinite, abs(offset) < Double(Int.max) else { throw ReplayError("Invalid capture action time.") }
            let tick = Int(offset.rounded(.toNearestOrEven))
            if tick < 0 { continue }
            guard byTick[tick] == nil else { throw ReplayError("Multiple impulses occupy one simulation tick.") }
            byTick[tick] = action
        }
        var model = try MetroRunnerModel(config: MetroRunnerConfig(frameSkip: 1, maxEpisodeSeconds: duration + 1))
        _ = try model.reset(seed: 7, options: nil)
        model.entities = []
        model.nextChunk = .infinity
        model.motor.lane = lane
        model.motor.x = Double(lane)
        model.motor.laneFrom = Double(lane)
        model.motor.laneTo = Double(lane)
        model.motor.z = first.z
        model.motor.y = first.y
        model.motor.elapsed = first.t
        model.surface = first.y
        model.motor.ground = first.y
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        var result = Data()
        func append<T: Encodable>(_ row: T) throws {
            result.append(try encoder.encode(row))
            result.append(10)
        }
        var candidateHeader = header
        candidateHeader.source = "simulator"
        candidateHeader.fixture = "native Swift; empty track; default character; no powers; original first sample"
        try append(candidateHeader)
        try append(Sample(model))
        for tick in 0..<Int(ceil(duration * 60)) {
            if let action = byTick[tick] {
                try append(Action(t: model.motor.elapsed, action: action.action))
            }
            _ = try model.step(byTick[tick].flatMap { commands[$0.action] }?.rawValue ?? 0)
            try append(Sample(model))
        }
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        try result.write(to: output, options: .withoutOverwriting)
    }

    private struct Kind: Decodable { let kind: String }

    private struct Header: Codable {
        var kind: String
        let schemaVersion: Int
        var source: String
        let targetUUID: String
        let scenario: String
        var fixture: String?

        enum CodingKeys: String, CodingKey {
            case kind, source, scenario, fixture
            case schemaVersion = "schema_version"
            case targetUUID = "target_uuid"
        }
    }

    private struct Sample: Codable {
        var kind = "sample"
        let t: Double
        let x: Double
        let y: Double
        let z: Double
        let vy: Double
        let vz: Double
        let grounded: Bool
        let flying: Bool

        init(_ model: MetroRunnerModel) {
            t = model.motor.elapsed; x = model.motor.worldX; y = model.motor.y; z = model.motor.z
            vy = model.motor.vy; vz = model.speed; grounded = model.motor.grounded; flying = model.power(.jetpack) > 0
        }
    }

    private struct Action: Codable {
        var kind = "action"
        let t: Double
        let action: String
    }

    private struct ReplayError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }
}
