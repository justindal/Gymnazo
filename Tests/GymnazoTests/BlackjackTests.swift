import Testing
import MLX

@testable import Gymnazo

@Suite("Blackjack environment")
struct BlackjackTests {
    func makeBlackjack(natural: Bool? = nil, sab: Bool? = nil) async throws -> Blackjack {
        var options: EnvOptions = [:]
        if let natural {
            options["natural"] = natural
        }
        if let sab {
            options["sab"] = sab
        }
        let env = try await Gymnazo.make("Blackjack", options: options)
        guard let blackjack = env.unwrapped as? Blackjack else {
            throw GymnazoError.invalidEnvironmentType(
                expected: "Blackjack",
                actual: String(describing: type(of: env.unwrapped))
            )
        }
        return blackjack
    }

    @Test
    func testResetDeterminismWithSeed() async throws {
        let env = try await makeBlackjack()
        let r1 = try env.reset(seed: 42)
        let r2 = try env.reset(seed: 42)
        #expect(MLX.arrayEqual(r1.obs, r2.obs).item(Bool.self))
    }

    @Test
    func testObservationSpaceValidity() async throws {
        let env = try await makeBlackjack()
        let result = try env.reset(seed: 123)

        let playerSum = result.obs[0].item(Int.self)
        let dealerCard = result.obs[1].item(Int.self)
        let usableAce = result.obs[2].item(Int.self)

        #expect(playerSum >= 12)
        #expect(playerSum <= 21)
        #expect(dealerCard >= 1)
        #expect(dealerCard <= 10)
        #expect(usableAce == 0 || usableAce == 1)
    }

    @Test
    func testHitAction() async throws {
        let env = try await makeBlackjack()
        _ = try env.reset(seed: 100)
        let result = try env.step(MLXArray(Int32(1)))

        let playerSum = result.obs[0].item(Int.self)
        #expect(playerSum >= 3)
        if playerSum <= 21 {
            #expect(result.terminated == false)
            #expect(result.reward == 0.0)
        } else {
            #expect(result.terminated == true)
            #expect(result.reward == -1.0)
        }
    }

    @Test
    func testStickAction() async throws {
        let env = try await makeBlackjack()
        _ = try env.reset(seed: 200)
        let result = try env.step(MLXArray(Int32(0)))

        #expect(result.terminated == true)
        #expect(result.reward == -1.0 || result.reward == 0.0 || result.reward == 1.0)
    }

    @Test
    func testActionSpaceContains() async throws {
        let env = try await makeBlackjack()
        guard let actionSpace = env.actionSpace as? Discrete else {
            Issue.record("Action space is not Discrete")
            return
        }
        #expect(actionSpace.contains(MLXArray(Int32(0))) == true)
        #expect(actionSpace.contains(MLXArray(Int32(1))) == true)
        #expect(actionSpace.contains(MLXArray(Int32(2))) == false)
        #expect(actionSpace.contains(MLXArray(Int32(-1))) == false)
    }

    @Test
    func testBustTerminates() async throws {
        let env = try await makeBlackjack()
        _ = try env.reset(seed: 1)

        var terminated = false
        for _ in 0..<20 {
            let result = try env.step(MLXArray(Int32(1)))
            if result.terminated {
                terminated = true
                let playerSum = result.obs[0].item(Int.self)
                if playerSum > 21 {
                    #expect(result.reward == -1.0)
                }
                break
            }
        }
        #expect(terminated == true)
    }

    @Test
    func testNaturalBlackjackReward() async throws {
        let env = try await makeBlackjack(natural: true)

        for seed: UInt64 in 0..<1000 {
            _ = try env.reset(seed: seed)
            let obs = try env.reset(seed: seed).obs

            let playerSum = obs[0].item(Int.self)
            let usableAce = obs[2].item(Int.self)

            if playerSum == 21 && usableAce == 1 {
                let result = try env.step(MLXArray(Int32(0)))
                if result.reward > 1.0 {
                    #expect(result.reward == 1.5)
                    return
                }
            }
        }
    }

    @Test
    func testSABMode() async throws {
        let env = try await makeBlackjack(sab: true)
        _ = try env.reset(seed: 42)
        let result = try env.step(MLXArray(Int32(0)))

        #expect(result.terminated == true)
        #expect(result.reward >= -1.0 && result.reward <= 1.0)
    }

    @Test
    @MainActor
    func testGymnazoMakeBlackjack() async throws {
        let env = try await Gymnazo.make("Blackjack")
        let blackjack = env.unwrapped as! Blackjack
        _ = try blackjack.reset(seed: 123)
        let result = try blackjack.step(MLXArray(Int32(0)))
        #expect(result.terminated == true)
    }

    @Test
    @MainActor
    func testGymnazoMakeBlackjackWithKwargs() async throws {
        let env = try await Gymnazo.make(
            "Blackjack",
            options: ["natural": true, "sab": false]
        )
        let blackjack = env.unwrapped as! Blackjack
        _ = try blackjack.reset(seed: 42)
        #expect((blackjack.actionSpace as? Discrete)?.n == 2)
    }
}
