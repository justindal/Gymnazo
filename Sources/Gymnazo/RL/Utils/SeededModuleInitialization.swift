import MLX
import MLXNN

func initializeModuleWeights(_ module: Module, key: inout MLXArray) throws {
    if let linear = module as? Linear {
        let (weightKey, nextKey) = MLX.split(key: key, stream: .cpu)
        key = nextKey
        let scale = Float(1.0 / Float(linear.weight.shape[1])).squareRoot()
        var parameters: [String: MLXArray] = [
            "weight": MLX.uniform(-scale..<scale, linear.weight.shape, key: weightKey)
        ]
        if linear.bias != nil {
            let (biasKey, nextKey) = MLX.split(key: key, stream: .cpu)
            key = nextKey
            parameters["bias"] = MLX.uniform(-scale..<scale, [linear.weight.shape[0]], key: biasKey)
        }
        try linear.update(parameters: ModuleParameters.unflattened(parameters), verify: .none)
    } else if let convolution = module as? Conv1d {
        try initializeConvolutionWeights(convolution, key: &key)
    } else if let convolution = module as? Conv2d {
        try initializeConvolutionWeights(convolution, key: &key)
    } else if let convolution = module as? Conv3d {
        try initializeConvolutionWeights(convolution, key: &key)
    }

    let children = module.children()
    for name in children.keys.sorted() {
        guard let child = children[name] else { continue }
        try initializeModuleWeights(child: child, key: &key)
    }
}

private func initializeConvolutionWeights(_ convolution: Module, key: inout MLXArray) throws {
    let weight: MLXArray
    let bias: MLXArray?
    let groups: Int
    switch convolution {
    case let convolution as Conv1d:
        weight = convolution.weight
        bias = convolution.bias
        groups = convolution.groups
    case let convolution as Conv2d:
        weight = convolution.weight
        bias = convolution.bias
        groups = convolution.groups
    case let convolution as Conv3d:
        weight = convolution.weight
        bias = convolution.bias
        groups = convolution.groups
    default:
        return
    }

    let (weightKey, nextKey) = MLX.split(key: key, stream: .cpu)
    key = nextKey
    let fanIn = weight.shape.dropFirst().reduce(groups, *)
    let scale = Float(1.0 / Float(fanIn)).squareRoot()
    var parameters: [String: MLXArray] = [
        "weight": MLX.uniform(-scale..<scale, weight.shape, key: weightKey)
    ]
    if bias != nil {
        parameters["bias"] = MLX.zeros([weight.shape[0]])
    }
    try convolution.update(parameters: ModuleParameters.unflattened(parameters), verify: .none)
}

private func initializeModuleWeights(child: NestedItem<String, Module>, key: inout MLXArray) throws
{
    switch child {
    case .none:
        return
    case .value(let module):
        try initializeModuleWeights(module, key: &key)
    case .array(let modules):
        for module in modules {
            try initializeModuleWeights(child: module, key: &key)
        }
    case .dictionary(let modules):
        for name in modules.keys.sorted() {
            guard let module = modules[name] else { continue }
            try initializeModuleWeights(child: module, key: &key)
        }
    }
}
