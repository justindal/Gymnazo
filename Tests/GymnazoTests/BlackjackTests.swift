import Testing
@testable import Gymnazo

@Suite("Blackjack environment")
struct BlackjackTests {
    @Test
    func testResetDeterminismWithSeed() async throws {
        let env = Blackjack()
        let r1 = env.reset(seed: 42)
        let r2 = env.reset(seed: 42)
        #expect(r1.obs == r2.obs)
    }
    
    @Test
    func testObservationSpaceValidity() async throws {
        let env = Blackjack()
        let result = env.reset(seed: 123)
        
        #expect(result.obs.playerSum >= 12)
        #expect(result.obs.playerSum <= 21)
        #expect(result.obs.dealerCard >= 1)
        #expect(result.obs.dealerCard <= 10)
        #expect(result.obs.usableAce == 0 || result.obs.usableAce == 1)
    }
    
    @Test
    func testHitAction() async throws {
        let env = Blackjack()
        _ = env.reset(seed: 100)
        let result = env.step(1)
        
        #expect(result.obs.playerSum >= 3)
        if result.obs.playerSum <= 21 {
            #expect(result.terminated == false)
            #expect(result.reward == 0.0)
        } else {
            #expect(result.terminated == true)
            #expect(result.reward == -1.0)
        }
    }
    
    @Test
    func testStickAction() async throws {
        let env = Blackjack()
        _ = env.reset(seed: 200)
        let result = env.step(0)
        
        #expect(result.terminated == true)
        #expect(result.reward == -1.0 || result.reward == 0.0 || result.reward == 1.0)
    }
    
    @Test
    func testActionSpaceContains() async throws {
        let env = Blackjack()
        #expect(env.action_space.contains(0) == true)
        #expect(env.action_space.contains(1) == true)
        #expect(env.action_space.contains(2) == false)
        #expect(env.action_space.contains(-1) == false)
    }
    
    @Test
    func testBustTerminates() async throws {
        let env = Blackjack()
        _ = env.reset(seed: 1)
        
        var terminated = false
        for _ in 0..<20 {
            let result = env.step(1)
            if result.terminated {
                terminated = true
                if result.obs.playerSum > 21 {
                    #expect(result.reward == -1.0)
                }
                break
            }
        }
        #expect(terminated == true)
    }
    
    @Test
    func testNaturalBlackjackReward() async throws {
        let env = Blackjack(natural: true)
        
        for seed: UInt64 in 0..<1000 {
            _ = env.reset(seed: seed)
            let obs = env.reset(seed: seed).obs
            
            if obs.playerSum == 21 && obs.usableAce == 1 {
                let result = env.step(0)
                if result.reward > 1.0 {
                    #expect(result.reward == 1.5)
                    return
                }
            }
        }
    }
    
    @Test
    func testSABMode() async throws {
        let env = Blackjack(sab: true)
        _ = env.reset(seed: 42)
        let result = env.step(0)
        
        #expect(result.terminated == true)
        #expect(result.reward >= -1.0 && result.reward <= 1.0)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeBlackjack() async throws {
        let env = Gymnazo.make("Blackjack")
        let blackjack = env.unwrapped as! Blackjack
        _ = blackjack.reset(seed: 123)
        let result = blackjack.step(0)
        #expect(result.terminated == true)
    }
    
    @Test
    @MainActor
    func testGymnazoMakeBlackjackWithKwargs() async throws {
        let env = Gymnazo.make("Blackjack", kwargs: ["natural": true, "sab": false])
        let blackjack = env.unwrapped as! Blackjack
        _ = blackjack.reset(seed: 42)
        #expect(blackjack.action_space.n == 2)
    }
}

