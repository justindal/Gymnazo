//
//  ProgressCallback.swift
//  Gymnazo
//

import Foundation

/// Callback that displays training progress.
public final class ProgressCallback: Callback {
    public let logInterval: Int
    public var logger: (any Logger)?

    private var startTime: Date?
    private var lastLogTimestep: Int = 0
    private var lastLogTime: Date?
    private var episodeMonitor: EpisodeMonitor?

    /// Creates a progress callback.
    ///
    /// - Parameters:
    ///   - logInterval: Log progress every N timesteps.
    ///   - logger: Logger for recording metrics.
    ///   - episodeMonitor: Episode monitor for reward tracking.
    public init(
        logInterval: Int = 1000,
        logger: (any Logger)? = nil,
        episodeMonitor: EpisodeMonitor? = nil
    ) {
        self.logInterval = logInterval
        self.logger = logger
        self.episodeMonitor = episodeMonitor
    }

    /// Sets the episode monitor for reward tracking.
    ///
    /// - Parameter monitor: The episode monitor to use.
    public func setEpisodeMonitor(_ monitor: EpisodeMonitor) {
        self.episodeMonitor = monitor
    }

    public func onTrainingStart(locals: CallbackLocals) {
        startTime = Date()
        lastLogTime = startTime
        lastLogTimestep = 0
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        if locals.numTimesteps - lastLogTimestep < logInterval {
            return true
        }

        let now = Date()
        let totalElapsed = startTime.map { now.timeIntervalSince($0) } ?? 0
        let intervalElapsed = lastLogTime.map { now.timeIntervalSince($0) } ?? 0
        let intervalSteps = locals.numTimesteps - lastLogTimestep

        let fps = intervalElapsed > 0 ? Double(intervalSteps) / intervalElapsed : 0
        let progress = Double(locals.numTimesteps) / Double(max(locals.totalTimesteps, 1)) * 100

        logger?.record(LogKey.Time.fps, value: fps)
        logger?.record(LogKey.Time.totalTimesteps, value: Double(locals.numTimesteps))
        logger?.record(LogKey.Time.timeElapsed, value: totalElapsed)
        logger?.record(LogKey.Time.iterations, value: Double(locals.iteration))

        if let monitor = episodeMonitor, monitor.windowCount > 0 {
            logger?.record(LogKey.Rollout.epRewMean, value: monitor.meanReward)
            logger?.record(LogKey.Rollout.epRewStd, value: monitor.stdReward)
            logger?.record(LogKey.Rollout.epLenMean, value: monitor.meanLength)
        }

        logger?.dump(step: locals.numTimesteps)

        printProgress(
            timesteps: locals.numTimesteps,
            totalTimesteps: locals.totalTimesteps,
            progress: progress,
            fps: fps,
            elapsed: totalElapsed
        )

        lastLogTimestep = locals.numTimesteps
        lastLogTime = now

        return true
    }

    public func onTrainingEnd(locals: CallbackLocals) {
        guard let start = startTime else { return }
        let elapsed = Date().timeIntervalSince(start)
        let avgFps = elapsed > 0 ? Double(locals.numTimesteps) / elapsed : 0
        print(
            "Training completed: \(locals.numTimesteps) timesteps in \(formatDuration(elapsed)) (avg \(String(format: "%.0f", avgFps)) fps)"
        )
    }

    private func printProgress(
        timesteps: Int,
        totalTimesteps: Int,
        progress: Double,
        fps: Double,
        elapsed: TimeInterval
    ) {
        let bar = makeProgressBar(progress: progress / 100, width: 20)
        var line =
            "\r[\(bar)] \(String(format: "%5.1f", progress))% | \(timesteps)/\(totalTimesteps) | \(String(format: "%.0f", fps)) fps"

        if let monitor = episodeMonitor, monitor.windowCount > 0 {
            line += " | reward: \(String(format: "%.2f", monitor.meanReward))"
        }

        print(line, terminator: "")
        fflush(stdout)
    }

    private func makeProgressBar(progress: Double, width: Int) -> String {
        let filled = Int(progress * Double(width))
        let empty = width - filled
        return String(repeating: "=", count: filled) + String(repeating: " ", count: empty)
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.1fs", seconds)
        } else if seconds < 3600 {
            let mins = Int(seconds) / 60
            let secs = Int(seconds) % 60
            return "\(mins)m \(secs)s"
        } else {
            let hours = Int(seconds) / 3600
            let mins = (Int(seconds) % 3600) / 60
            return "\(hours)h \(mins)m"
        }
    }
}

/// Callback that logs to the console at regular intervals.
public final class ConsoleLogCallback: Callback {
    public let logInterval: Int
    public let metrics: [(key: String, format: String)]

    private var lastLogTimestep: Int = 0
    private let valueProviders: [String: () -> Double?]

    /// Creates a console log callback.
    ///
    /// - Parameters:
    ///   - logInterval: Log every N timesteps.
    ///   - metrics: List of (key, format) pairs to log.
    ///   - valueProviders: Dictionary mapping keys to value provider closures.
    public init(
        logInterval: Int = 1000,
        metrics: [(key: String, format: String)] = [],
        valueProviders: [String: () -> Double?] = [:]
    ) {
        self.logInterval = logInterval
        self.metrics = metrics
        self.valueProviders = valueProviders
    }

    public func onStep(locals: CallbackLocals) -> Bool {
        if locals.numTimesteps - lastLogTimestep < logInterval {
            return true
        }

        lastLogTimestep = locals.numTimesteps

        var parts: [String] = ["step: \(locals.numTimesteps)"]

        for (key, format) in metrics {
            if let provider = valueProviders[key], let value = provider() {
                parts.append("\(key): \(String(format: format, value))")
            }
        }

        print(parts.joined(separator: " | "))

        return true
    }
}
