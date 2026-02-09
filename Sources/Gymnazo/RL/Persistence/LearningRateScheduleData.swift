//
//  LearningRateScheduleData.swift
//  Gymnazo
//

import Foundation

/// Serializable representation of a learning rate schedule.
public enum LearningRateScheduleData: Codable, Sendable {
    case constant(value: Double)
    case linear(initialValue: Double, finalValue: Double)
    case exponential(initialValue: Double, decayRate: Double)
    case step(initialValue: Double, milestones: [Double], gamma: Double)
    case cosineAnnealing(initialValue: Double, minValue: Double)
    indirect case warmup(
        base: LearningRateScheduleData,
        warmupFraction: Double,
        warmupInitialValue: Double
    )

    public func makeSchedule() -> any LearningRateSchedule {
        switch self {
        case .constant(let value):
            return ConstantLearningRate(value)
        case .linear(let initialValue, let finalValue):
            return LinearSchedule(initialValue: initialValue, finalValue: finalValue)
        case .exponential(let initialValue, let decayRate):
            return ExponentialSchedule(initialValue: initialValue, decayRate: decayRate)
        case .step(let initialValue, let milestones, let gamma):
            return StepSchedule(initialValue: initialValue, milestones: milestones, gamma: gamma)
        case .cosineAnnealing(let initialValue, let minValue):
            return CosineAnnealingSchedule(initialValue: initialValue, minValue: minValue)
        case .warmup(let base, let warmupFraction, let warmupInitialValue):
            return WarmupSchedule(
                baseSchedule: base.makeSchedule(),
                warmupFraction: warmupFraction,
                warmupInitialValue: warmupInitialValue
            )
        }
    }

    public static func from(_ schedule: any LearningRateSchedule) -> LearningRateScheduleData? {
        switch schedule {
        case let s as ConstantLearningRate:
            return .constant(value: s.value)
        case let s as LinearSchedule:
            return .linear(initialValue: s.initialValue, finalValue: s.finalValue)
        case let s as ExponentialSchedule:
            return .exponential(initialValue: s.initialValue, decayRate: s.decayRate)
        case let s as StepSchedule:
            return .step(initialValue: s.initialValue, milestones: s.milestones, gamma: s.gamma)
        case let s as CosineAnnealingSchedule:
            return .cosineAnnealing(initialValue: s.initialValue, minValue: s.minValue)
        case let s as WarmupSchedule:
            guard let baseData = from(s.baseSchedule) else { return nil }
            return .warmup(
                base: baseData,
                warmupFraction: s.warmupFraction,
                warmupInitialValue: s.warmupInitialValue
            )
        default:
            return nil
        }
    }
}
