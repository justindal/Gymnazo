import Testing
import MLX
@testable import Gymnazo

final class FrozenLakeInfoTagWrapper: Wrapper {
    var env: any Env
    let tagKey: String
    let tagValue: InfoValue
    
    init(env: any Env) {
        self.env = env
        self.tagKey = "wrapped"
        self.tagValue = .bool(true)
    }
    
    init(env: any Env, tagKey: String, tagValue: InfoValue) {
        self.env = env
        self.tagKey = tagKey
        self.tagValue = tagValue
    }
    
    func step(_ action: MLXArray) throws -> Step {
        let r = try env.step(action)
        var info = r.info
        info[tagKey] = tagValue
        return Step(obs: r.obs, reward: r.reward, terminated: r.terminated, truncated: r.truncated, info: info)
    }
}

@Suite("Additional wrapper application")
struct AdditionalWrappersTests {
    @Test
    @MainActor
    func testAdditionalWrapperApplied() async throws {
        let _ = try await Gymnazo.make(
            "FrozenLake",
            options: ["is_slippery": false]
        )

        let specs = await Gymnazo.registry()
        guard var baseSpec = specs["FrozenLake"] else {
            #expect(Bool(false), "FrozenLake spec missing")
            return
        }
        
        let wrapperSpec = WrapperSpec(
            id: "info-tag",
            entryPoint: { env, kwargs in
                let key = (kwargs["key"] as? String) ?? "wrapped"
                let rawValue = kwargs["value"]
                let value: InfoValue
                if let b = rawValue as? Bool {
                    value = .bool(b)
                } else if let i = rawValue as? Int {
                    value = .int(i)
                } else if let d = rawValue as? Double {
                    value = .double(d)
                } else if let f = rawValue as? Float {
                    value = .double(Double(f))
                } else if let s = rawValue as? String {
                    value = .string(s)
                } else {
                    value = .bool(true)
                }
                return FrozenLakeInfoTagWrapper(env: env, tagKey: key, tagValue: value)
            },
            options: ["key": "extra", "value": "ok"]
        )
        
        baseSpec.additionalWrappers = [wrapperSpec]
        
        let env = try await Gymnazo.make(
            baseSpec,
            recordEpisodeStatistics: false,
            options: ["is_slippery": false]
        )
        
        let hasWrapper = env.spec?.additionalWrappers.contains(where: { $0.id == "info-tag" }) ?? false
        #expect(hasWrapper)
    }
}
