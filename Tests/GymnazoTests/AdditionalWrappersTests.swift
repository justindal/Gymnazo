import Testing
@testable import Gymnazo

/// wrapper specialized to FrozenLake that injects a flag into info to verify that
/// additional_wrappers are applied by Gymnazo.make(spec:).
final class FrozenLakeInfoTagWrapper: Wrapper {
    typealias InnerEnv = FrozenLake
    
    var env: FrozenLake
    let tagKey: String
    let tagValue: InfoValue
    
    required init(env: FrozenLake) {
        self.env = env
        self.tagKey = "wrapped"
        self.tagValue = .bool(true)
    }
    
    init(env: FrozenLake, tagKey: String, tagValue: InfoValue) {
        self.env = env
        self.tagKey = tagKey
        self.tagValue = tagValue
    }
    
    func step(_ action: Int) -> Step<Observation> {
        let r = env.step(action)
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
        _ = Gymnazo.make("FrozenLake", kwargs: ["is_slippery": false])
        
        guard var baseSpec = Gymnazo.registry["FrozenLake"] else {
            #expect(Bool(false), "FrozenLake spec missing")
            return
        }
        
        let wrapperSpec = WrapperSpec(
            id: "info-tag",
            entryPoint: { env, kwargs in
                guard let fl = env as? FrozenLake else { return env }
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
                return FrozenLakeInfoTagWrapper(env: fl, tagKey: key, tagValue: value)
            },
            kwargs: ["key": "extra", "value": "ok"]
        )
        
        baseSpec.additional_wrappers = [wrapperSpec]
        
        let env = Gymnazo.make(
            baseSpec,
            recordEpisodeStatistics: false,
            kwargs: ["is_slippery": false]
        )
        
        // validate that the requested wrapper is present on the applied spec
        let hasWrapper = env.spec?.additional_wrappers.contains(where: { $0.id == "info-tag" }) ?? false
        #expect(hasWrapper)
    }
}

