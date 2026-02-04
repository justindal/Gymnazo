//
//  Logger.swift
//  Gymnazo
//

import Foundation

/// Namespaced keys for consistent logging across algorithms.
public enum LogKey {
    public enum Rollout {
        public static let epRewMean = "rollout/ep_rew_mean"
        public static let epRewStd = "rollout/ep_rew_std"
        public static let epLenMean = "rollout/ep_len_mean"
        public static let successRate = "rollout/success_rate"
    }

    public enum Train {
        public static let loss = "train/loss"
        public static let policyLoss = "train/policy_loss"
        public static let valueLoss = "train/value_loss"
        public static let entropyLoss = "train/entropy_loss"
        public static let learningRate = "train/learning_rate"
        public static let nUpdates = "train/n_updates"
        public static let approxKL = "train/approx_kl"
        public static let clipFraction = "train/clip_fraction"
        public static let entCoef = "train/ent_coef"
        public static let actorLoss = "train/actor_loss"
        public static let criticLoss = "train/critic_loss"
        public static let explorationRate = "train/exploration_rate"
    }

    public enum Eval {
        public static let meanReward = "eval/mean_reward"
        public static let stdReward = "eval/std_reward"
        public static let meanEpLength = "eval/mean_ep_length"
    }

    public enum Time {
        public static let fps = "time/fps"
        public static let totalTimesteps = "time/total_timesteps"
        public static let timeElapsed = "time/time_elapsed"
        public static let iterations = "time/iterations"
    }
}

/// Output format for logging.
public enum LogFormat: Sendable {
    case stdout
    case csv(URL)
    case json(URL)
}

/// Protocol for logging training metrics.
public protocol Logger: AnyObject {
    /// Records a scalar value.
    ///
    /// - Parameters:
    ///   - key: The metric key.
    ///   - value: The scalar value.
    func record(_ key: String, value: Double)

    /// Records a string value.
    ///
    /// - Parameters:
    ///   - key: The metric key.
    ///   - value: The string value.
    func record(_ key: String, value: String)

    /// Writes all recorded values to configured outputs.
    ///
    /// - Parameter step: The current timestep.
    func dump(step: Int)

    /// Retrieves the latest recorded value for a key.
    ///
    /// - Parameter key: The metric key.
    /// - Returns: The latest value, or nil if not recorded.
    func getLatest(_ key: String) -> Double?
}

/// Standard logger implementation supporting multiple output formats.
public final class StandardLogger: Logger {
    private var outputs: [LogFormat]
    private var currentValues: [String: LogValue] = [:]
    private var history: [String: [(step: Int, value: Double)]] = [:]
    private var csvHeaders: Set<String> = []
    private var csvInitialized: [URL: Bool] = [:]

    private enum LogValue {
        case scalar(Double)
        case string(String)
    }

    /// Creates a standard logger with specified outputs.
    ///
    /// - Parameter outputs: The output formats to use.
    public init(outputs: [LogFormat] = [.stdout]) {
        self.outputs = outputs
    }

    public func record(_ key: String, value: Double) {
        currentValues[key] = .scalar(value)
    }

    public func record(_ key: String, value: String) {
        currentValues[key] = .string(value)
    }

    public func dump(step: Int) {
        for output in outputs {
            switch output {
            case .stdout:
                printToStdout(step: step)
            case .csv(let url):
                appendToCSV(url: url, step: step)
            case .json(let url):
                appendToJSON(url: url, step: step)
            }
        }

        for (key, value) in currentValues {
            if case .scalar(let v) = value {
                history[key, default: []].append((step, v))
            }
        }

        currentValues.removeAll()
    }

    public func getLatest(_ key: String) -> Double? {
        if let value = currentValues[key], case .scalar(let v) = value {
            return v
        }
        return history[key]?.last?.value
    }

    /// Returns the full history for a given key.
    ///
    /// - Parameter key: The metric key.
    /// - Returns: Array of (step, value) tuples.
    public func getHistory(_ key: String) -> [(step: Int, value: Double)] {
        history[key] ?? []
    }

    private func printToStdout(step: Int) {
        var lines: [String] = []
        lines.append("-" * 40)
        lines.append(
            "| \("timestep".padding(toLength: 20, withPad: " ", startingAt: 0)) | \(String(step).padding(toLength: 14, withPad: " ", startingAt: 0)) |"
        )

        let sortedKeys = currentValues.keys.sorted()
        for key in sortedKeys {
            guard let value = currentValues[key] else { continue }
            let displayKey = key.padding(toLength: 20, withPad: " ", startingAt: 0)
            let displayValue: String
            switch value {
            case .scalar(let v):
                displayValue = formatNumber(v).padding(toLength: 14, withPad: " ", startingAt: 0)
            case .string(let s):
                displayValue = s.padding(toLength: 14, withPad: " ", startingAt: 0)
            }
            lines.append("| \(displayKey) | \(displayValue) |")
        }

        lines.append("-" * 40)
        print(lines.joined(separator: "\n"))
    }

    private func appendToCSV(url: URL, step: Int) {
        var scalarValues: [(String, Double)] = []
        for (key, value) in currentValues {
            if case .scalar(let v) = value {
                scalarValues.append((key, v))
            }
        }

        scalarValues.sort { $0.0 < $1.0 }

        let isNew = csvInitialized[url] != true

        if isNew {
            let headerKeys = scalarValues.map { $0.0 }
            csvHeaders = Set(headerKeys)
            let header = "step," + headerKeys.joined(separator: ",") + "\n"

            do {
                try header.write(to: url, atomically: true, encoding: .utf8)
                csvInitialized[url] = true
            } catch {
                return
            }
        }

        let row = "\(step)," + scalarValues.map { formatNumber($0.1) }.joined(separator: ",") + "\n"

        do {
            let handle = try FileHandle(forWritingTo: url)
            handle.seekToEndOfFile()
            if let data = row.data(using: .utf8) {
                handle.write(data)
            }
            try handle.close()
        } catch {
            return
        }
    }

    private func appendToJSON(url: URL, step: Int) {
        var entry: [String: Any] = ["step": step]
        for (key, value) in currentValues {
            switch value {
            case .scalar(let v):
                entry[key] = v
            case .string(let s):
                entry[key] = s
            }
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: entry)
            var content = String(data: data, encoding: .utf8) ?? "{}"
            content += "\n"

            if FileManager.default.fileExists(atPath: url.path) {
                let handle = try FileHandle(forWritingTo: url)
                handle.seekToEndOfFile()
                if let lineData = content.data(using: .utf8) {
                    handle.write(lineData)
                }
                try handle.close()
            } else {
                try content.write(to: url, atomically: true, encoding: .utf8)
            }
        } catch {
            return
        }
    }

    private func formatNumber(_ value: Double) -> String {
        if abs(value) >= 1e6 || (abs(value) < 1e-3 && value != 0) {
            return String(format: "%.3e", value)
        } else if value == floor(value) && abs(value) < 1e6 {
            return String(format: "%.0f", value)
        } else {
            return String(format: "%.4f", value)
        }
    }
}

extension String {
    fileprivate static func * (lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
