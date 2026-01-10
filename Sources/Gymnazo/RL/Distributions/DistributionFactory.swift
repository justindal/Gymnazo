//
//  DistributionFactory.swift
//  Gymnazo
//

import MLX
import MLXNN

/// Configuration for State-Dependent Exploration (gSDE).
public struct SDEConfig {
    public let fullStd: Bool
    public let squashOutput: Bool
    public let learnFeatures: Bool
    
    public init(
        fullStd: Bool = true,
        squashOutput: Bool = false,
        learnFeatures: Bool = false
    ) {
        self.fullStd = fullStd
        self.squashOutput = squashOutput
        self.learnFeatures = learnFeatures
    }
}

/// Enum representing different distribution types.
public enum DistributionType {
    case diagGaussian(actionDim: Int)
    case categorical(actionDim: Int)
    case multiCategorical(actionDims: [Int])
    case bernoulli(actionDim: Int)
    case stateDependentNoise(actionDim: Int, config: SDEConfig)
}

/// Factory for creating action distributions based on action space.
public enum DistributionFactory {
    /// Creates an appropriate distribution for the given action space.
    ///
    /// - Parameters:
    ///   - actionSpace: The action space of the environment.
    ///   - useSDE: Whether to use State-Dependent Exploration.
    ///   - sdeConfig: Configuration for gSDE (if useSDE is true).
    /// - Returns: A distribution matching the action space.
    public static func makeProbaDistribution(
        actionSpace: any Space,
        useSDE: Bool = false,
        sdeConfig: SDEConfig? = nil
    ) -> any Distribution {
        if let box = actionSpace as? Box {
            let actionDim = box.shape?.reduce(1, *) ?? 1
            
            if useSDE {
                let config = sdeConfig ?? SDEConfig()
                return StateDependentNoiseDistribution(
                    actionDim: actionDim,
                    fullStd: config.fullStd,
                    squashOutput: config.squashOutput,
                    learnFeatures: config.learnFeatures
                )
            }
            
            return DiagGaussianDistribution(actionDim: actionDim)
        }
        
        if let discrete = actionSpace as? Discrete {
            return CategoricalDistribution(actionDim: discrete.n)
        }
        
        if let multiDiscrete = actionSpace as? MultiDiscrete {
            let dims = multiDiscrete.nvec.asArray(Int32.self).map(Int.init)
            return MultiCategoricalDistribution(actionDims: dims)
        }
        
        if let multiBinary = actionSpace as? MultiBinary {
            let actionDim = multiBinary.shape?.reduce(1, *) ?? 1
            return BernoulliDistribution(actionDim: actionDim)
        }
        
        fatalError("Unsupported action space type: \(type(of: actionSpace))")
    }
    
    /// Creates a distribution from an explicit type specification.
    ///
    /// - Parameter distType: The distribution type to create.
    /// - Returns: The created distribution.
    public static func makeDistribution(_ distType: DistributionType) -> any Distribution {
        switch distType {
        case .diagGaussian(let actionDim):
            return DiagGaussianDistribution(actionDim: actionDim)
            
        case .categorical(let actionDim):
            return CategoricalDistribution(actionDim: actionDim)
            
        case .multiCategorical(let actionDims):
            return MultiCategoricalDistribution(actionDims: actionDims)
            
        case .bernoulli(let actionDim):
            return BernoulliDistribution(actionDim: actionDim)
            
        case .stateDependentNoise(let actionDim, let config):
            return StateDependentNoiseDistribution(
                actionDim: actionDim,
                fullStd: config.fullStd,
                squashOutput: config.squashOutput,
                learnFeatures: config.learnFeatures
            )
        }
    }
}

