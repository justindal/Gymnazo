import MLX
import MLXOptimizers

final class ScheduledAdam: Adam {
    private var rate: MLXArray
    private var capturedRate: Float

    init(learningRate: Float, betas: (Float, Float) = (0.9, 0.999), eps: Float = 1e-8) {
        self.rate = MLXArray(learningRate)
        self.capturedRate = learningRate
        super.init(learningRate: learningRate, betas: betas, eps: eps)
    }

    override func innerState() -> [MLXArray] {
        synchronizeLearningRate()
        return [rate] + super.innerState()
    }

    override func applySingle(
        gradient: MLXArray,
        parameter: MLXArray,
        state: AdamState
    ) -> (MLXArray, AdamState) {
        synchronizeLearningRate()
        let (_, next) = super.applySingle(gradient: gradient, parameter: parameter, state: state)
        let moments = next.innerState()
        let update: MLXArray
        if biasCorrection {
            let scale = rate / (1 - MLX.pow(betas.0, moments[2]))
            let correction = MLX.rsqrt(1 - MLX.pow(betas.1, moments[2]))
            update = scale * moments[0] / (MLX.sqrt(moments[1]) * correction + eps)
        } else {
            update = rate * moments[0] / (MLX.sqrt(moments[1]) + eps)
        }
        return (parameter - update, next)
    }

    private func synchronizeLearningRate() {
        if capturedRate != learningRate {
            rate = MLXArray(learningRate)
            capturedRate = learningRate
        }
    }
}

extension OptimizerConfig {
    func makeScheduled(learningRate: Float, schedule: (any LearningRateSchedule)? = nil) -> Adam {
        if schedule is ConstantLearningRate {
            return make(learningRate: learningRate)
        }
        switch self {
        case .adam(let beta1, let beta2, let eps):
            return ScheduledAdam(learningRate: learningRate, betas: (beta1, beta2), eps: eps)
        }
    }
}
