import Foundation
import Gymnazo

enum MetroRunnerReplayCommand {
    static func run(_ arguments: [String]) throws {
        guard arguments.count == 4, arguments[1] == "--flat-track", arguments[2] == "--output" else {
            throw CommandError()
        }
        try MetroRunnerMotionReplay.replay(original: URL(filePath: arguments[0]),
            output: URL(filePath: arguments[3]), flatTrack: true)
        print("Wrote native simulator motion trace to \(arguments[3])")
    }

    private struct CommandError: LocalizedError {
        var errorDescription: String? {
            "Usage: GymnazoProfiler metro-replay ORIGINAL.jsonl --flat-track --output SIMULATOR.jsonl"
        }
    }
}
