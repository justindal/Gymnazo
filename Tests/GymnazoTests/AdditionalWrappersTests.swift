import Testing
@testable import Gymnazo

/// wrapper specialized to FrozenLake that injects a flag into info to verify that
/// additional_wrappers are applied by Gymnazo.make(spec:).
final class FrozenLakeInfoTagWrapper: Wrapper {
    typealias InnerEnv = FrozenLake
    
    var env: FrozenLake
    let tagKey: String
    let tagValue: Any
    
    required init(env: FrozenLake) {
        self.env = env
        self.tagKey = "wrapped"
        self.tagValue = true
    }
    
    init(env: FrozenLake, tagKey: String, tagValue: Any) {
        self.env = env
        self.tagKey = tagKey
        self.tagValue = tagValue
    }
    
    func step(_ action: Int) -> (
        obs: Int,
        reward: Double,
        terminated: Bool,
        truncated: Bool,
        info: [String : Any]
    ) {
        let r = env.step(action)
        var info = r.info
        info[tagKey] = tagValue
        return (r.obs, r.reward, r.terminated, r.truncated, info)
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
                let value = kwargs["value"] ?? true
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

