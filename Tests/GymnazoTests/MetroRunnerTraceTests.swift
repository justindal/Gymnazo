import Foundation
import Testing

@testable import Gymnazo

@Suite("Metro Run trace infrastructure", .serialized)
struct MetroRunnerTraceTests {
    private func fixture(_ url: URL, source: String = "original_game", times: [Double] = [0, 0.05, 0.1],
                         actions: [[String: Any]] = [], initialY: Double = 0) throws {
        let header: [String: Any] = ["kind": "header", "schema_version": 1, "source": source,
            "target_uuid": MetroRunnerMotionReplay.targetUUID, "scenario": "synthetic_codec_fixture"]
        let samples: [[String: Any]] = times.map { ["kind": "sample", "t": $0, "x": 0, "y": initialY,
            "z": 100 + $0 * 110, "vy": 0, "vz": 110, "grounded": true, "flying": false] }
        var result = Data()
        for row in [header] + samples + actions {
            result.append(try JSONSerialization.data(withJSONObject: row))
            result.append(10)
        }
        try result.write(to: url)
    }

    @Test
    func nativeReplayKeepsInitialPositionAndLabelsOutputAsSimulation() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let input = folder.appendingPathComponent("synthetic-input.jsonl"), output = folder.appendingPathComponent("simulator.jsonl")
        try fixture(input)
        try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true)
        let rows = try Data(contentsOf: output).split(separator: 10).map {
            try #require(JSONSerialization.jsonObject(with: Data($0)) as? [String: Any])
        }
        #expect(rows.count == 8)
        #expect(rows[0]["source"] as? String == "simulator")
        #expect(rows[0]["fixture"] != nil)
        #expect(rows[1]["z"] as? Double == 100)
        #expect(abs(try #require(rows.last?["t"] as? Double) - 0.1) < 1e-10)
        let existing = try Data(contentsOf: output)
        #expect(throws: (any Error).self) {
            try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true)
        }
        #expect(try Data(contentsOf: output) == existing)
    }

    @Test
    func invalidCaptureSourcesGapsAndConflictingInputsAreRejected() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let input = folder.appendingPathComponent("synthetic-input.jsonl"), output = folder.appendingPathComponent("simulator.jsonl")
        try fixture(input, source: "simulator")
        #expect(throws: (any Error).self) { try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true) }
        for times in [[0, 0, 0.1], [0, 0.2, 0.4]] {
            try fixture(input, times: times)
            #expect(throws: (any Error).self) { try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true) }
        }
        try fixture(input, initialY: 20)
        #expect(throws: (any Error).self) { try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true) }
        try fixture(input, actions: [["kind": "action", "t": 0.05, "action": "left"],
                                     ["kind": "action", "t": 0.051, "action": "right"]])
        #expect(throws: (any Error).self) { try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: true) }
        #expect(throws: (any Error).self) { try MetroRunnerMotionReplay.replay(original: input, output: output, flatTrack: false) }
        #expect(!FileManager.default.fileExists(atPath: output.path))
    }
}
