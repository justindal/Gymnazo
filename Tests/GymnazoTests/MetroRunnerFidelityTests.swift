import Foundation
import Testing

@testable import Gymnazo

@Suite("Metro Run recovered rules", .serialized)
struct MetroRunnerFidelityTests {
    private func empty(level: Int = 0) throws -> MetroRunnerModel {
        var model = try MetroRunnerModel(config: MetroRunnerConfig(frameSkip: 1, powerLevel: level))
        _ = try model.reset(seed: 7, options: nil)
        model.entities = []
        model.nextChunk = .infinity
        return model
    }

    @Test
    func defaultSpawnWeightsUnlocksAndInclusiveSpacing() throws {
        let rules = try MetroRunnerCatalog.bundled.get().rules
        var spawns = MetroRunnerSpawnRules(rules: rules, profile: .init(), generator: SeededGenerator(seed: 3))
        #expect(spawns.validEntries(at: 0, group: 1).map(\.weight) == [2, 29, 26, 14, 30, 29])
        #expect(spawns.choose(at: 0, group: 1, flying: false) != nil)
        #expect(spawns.choose(at: 2399.999, group: 1, flying: false) == nil)
        #expect(spawns.choose(at: 2400, group: 1, flying: false) != nil)
        #expect(spawns.validEntries(at: 10000, group: 2).isEmpty)
        #expect(spawns.validEntries(at: 10000, group: 3).isEmpty)
        var restricted = MetroRunnerSpawnRules(rules: rules, profile: .init(unlockedFeatures: [], lootBoxFriendly: false),
                                               generator: SeededGenerator(seed: 4))
        #expect(Set(restricted.validEntries(at: 0, group: 1).map(\.kind)) == [.key, .sneakers, .jetpack])
        restricted.lastPickupZ[.key] = 100
        #expect(!restricted.validEntries(at: 4899.99, group: 1).contains { $0.kind == .key })
        #expect(restricted.validEntries(at: 4900, group: 1).contains { $0.kind == .key })
        #expect(restricted.choose(at: 10000, group: 1, flying: true) == nil)
        #expect(restricted.lastGroupZ == nil)
    }

    @Test
    func longSeededPickupRunsRespectPerTypeDistances() throws {
        let rules = try MetroRunnerCatalog.bundled.get().rules
        var a = MetroRunnerSpawnRules(rules: rules, profile: .init(), generator: SeededGenerator(seed: 123))
        var b = a
        var previous: [MetroRunnerEntityKind: Double] = [:]
        var last = -Double.infinity
        for z in stride(from: 0.0, to: 100000, by: 100) {
            let kind = a.choose(at: z, group: 1, flying: false)
            #expect(kind == b.choose(at: z, group: 1, flying: false))
            if let kind {
                #expect(z - last >= 2400)
                let distances: [MetroRunnerEntityKind: Double] = [.key: 4800, .jetpack: 8000, .mystery: 57600]
                let minimum = distances[kind] ?? 0
                #expect(z - (previous[kind] ?? -.infinity) >= minimum)
                previous[kind] = z; last = z
            }
        }
        #expect(previous.count == 6)
    }

    @Test
    func exportedCoinCurvesControlCountsCenteringAndSpeedPivot() throws {
        let rules = try MetroRunnerCatalog.bundled.get().rules
        #expect(rules.coinSpawners["CoinLineSpawner_6_150"]?.count(speedModifier: 1) == 6)
        #expect(rules.coinPosition("CoinLineSpawner_6_150", t: 0, speedModifier: 1).z == -75)
        #expect(rules.coinPosition("CoinLineSpawner_6_150", t: 1, speedModifier: 1).z == 75)
        #expect(rules.coinSpawners["CoinLineSpawner_9_240"]?.count(speedModifier: 1) == 10)
        #expect(rules.coinSpawners["CoinCurveSpawner_7_scale0_5"]?.count(speedModifier: 2) == 14)
        #expect(abs(rules.coinPosition("CoinCurveSpawner_7_scale0_5", t: 0, speedModifier: 2).z + 100.2) < 1e-4)
        #expect(rules.coinPosition("CoinCurveSpawner_7_scale0_5", t: 0.5, speedModifier: 1).y > 19)
        #expect(abs(rules.coinPosition("CoinCurveSpawner_7_scale0_5", t: 1, speedModifier: 1).y + 0.07588) < 0.002)
        var model = try empty()
        model.entities = [MetroRunnerEntity(kind: .coin, x: 0, y: 0, z: 0,
            curveSpawner: "CoinCurveSpawner_7_scale0_5", curveT: 0,
            spawnOrigin: SIMD3(0, 6, 300))]
        model.motor.elapsed = 179
        model.updateCoinCurves()
        #expect(abs(model.entities[0].z - 199.8) < 1e-4)
    }

    @Test
    func coinDensityAndPowersDoNotConsumeRouteRandomness() throws {
        let catalog = try MetroRunnerCatalog.bundled.get()
        var a = try MetroRunnerRoute(catalog: catalog, seed: 9, section: nil), b = a
        for name in catalog.chunks.keys.sorted() {
            let (slow, _) = try a.instantiate(name, origin: 0, speed: 110)
            let (fast, _) = try b.instantiate(name, origin: 0, speed: 220)
            #expect(slow.filter { $0.kind.isHazard } == fast.filter { $0.kind.isHazard }.map { item in
                var item = item
                item.velocity /= 2
                return item
            })
        }
        let nextA = try a.next(at: 0), nextB = try b.next(at: 0)
        #expect(nextA == nextB)
    }

    @Test
    func allRestoredObstacleFamiliesUseExportedGeometry() throws {
        var route = try MetroRunnerRoute(catalog: MetroRunnerCatalog.bundled.get(), seed: 7, section: nil)
        var found: [MetroRunnerEntityKind: MetroRunnerEntity] = [:]
        var kinds: Set<MetroRunnerEntityKind> = []
        for _ in 0..<8 {
            for name in route.catalog.chunks.keys.sorted() {
                let (items, _) = try route.instantiate(name, origin: 0, speed: 110)
                kinds.formUnion(items.map(\.kind))
                for item in items where [.pillar, .signal, .dumpster, .bush, .powerbox].contains(item.kind) {
                    if found[item.kind] == nil { found[item.kind] = item }
                }
            }
        }
        #expect(found.count == 5)
        #expect(abs(try #require(found[.pillar]).height - 70) < 1e-5)
        #expect(abs(try #require(found[.signal]).width - 6) < 1e-5)
        #expect(abs(try #require(found[.dumpster]).height - 13.23859024) < 1e-5)
        #expect(abs(try #require(found[.bush]).height - 5) < 1e-5)
        #expect(abs(try #require(found[.powerbox]).height - 8.03337002) < 1e-5)
        #expect(kinds.contains(.pogo) && kinds.contains(.pickupAnchor))
        var model = try empty()
        _ = try model.reset(seed: 9, options: nil)
        #expect(!model.entities.contains { $0.kind == .pickupAnchor })
    }

    @Test
    func unknownObstacleAssetsFailCatalogValidation() throws {
        let url = try #require(Bundle.module.url(forResource: "metro-runner", withExtension: "json"))
        var json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var chunks = try #require(json["chunks"] as? [String: [String: Any]])
        var chunk = try #require(chunks["Chunk_Intro"])
        var root = try #require(chunk["root"] as? [String: Any])
        root["asset"] = "obstacle_unimplemented"
        chunk["root"] = root; chunks["Chunk_Intro"] = chunk; json["chunks"] = chunks
        let catalog = try JSONDecoder().decode(MetroRunnerCatalog.self, from: JSONSerialization.data(withJSONObject: json))
        #expect(throws: GymnazoError.self) { try catalog.validate() }
    }

    @Test
    func jumpGraceExpiresAirborneAndReversalScalesRemainingDistance() {
        var motor = MetroRunnerMotor()
        for _ in 0..<300 { motor.step(.none) }
        #expect(motor.sinceLanding == 0)
        motor.grounded = false; motor.y = 100
        for _ in 0..<5 { motor.step(.none) }
        #expect(motor.canJump)
        motor.step(.jump)
        #expect(motor.vy > 0 && !motor.canJump)
        motor.sinceLanding = 0.15
        #expect(!motor.canJump)
        motor = MetroRunnerMotor()
        motor.step(.right)
        for _ in 0..<4 { motor.step(.none) }
        let duration = 0.27 / (motor.speed / 110) * (motor.x - 1)
        motor.step(.left)
        #expect(abs(motor.laneDuration - duration) < 1e-12)
        for _ in 0..<20 { motor.step(.none) }
        #expect(motor.x == 1)
    }

    @Test
    func repeatedRollsAndAirDivesKeepCorrectColliderUntilLanding() throws {
        var model = try empty()
        for _ in 0..<30 { _ = try model.step(MetroRunnerAction.down.rawValue) }
        #expect(abs(model.motor.rolling - 0.1) < 1e-10)
        #expect(model.motor.bodyHeight == 4.5)
        model = try empty()
        model.motor.y = 20; model.motor.grounded = false; model.motor.sinceLanding = .infinity
        _ = try model.step(MetroRunnerAction.down.rawValue)
        #expect(model.motor.diving && model.motor.rolling == 0 && model.motor.bodyHeight == 9)
        let velocity = model.motor.vy
        _ = try model.step(MetroRunnerAction.down.rawValue)
        #expect(model.motor.vy < velocity)
        for _ in 0..<60 {
            _ = try model.step(0)
            if model.motor.grounded { break }
        }
        #expect(model.motor.grounded && !model.motor.diving && model.motor.rolling > 0.55)
    }

    @Test
    func roofContactRetainsTheMotorPhaseJumpGraceTimer() throws {
        var model = try empty()
        model.motor.y = 30.1
        model.motor.vy = -10
        model.motor.grounded = false
        model.motor.sinceLanding = 0.05
        model.entities = [MetroRunnerEntity(kind: .train, x: 9.5, y: 0, z: 0, width: 16, depth: 100, height: 30)]
        _ = try model.step(0)
        #expect(model.motor.grounded && model.motor.y == 30)
        #expect(abs(model.motor.sinceLanding - (0.05 + 1.0 / 60)) < 1e-10)
    }

    @Test
    func triggerObstaclesStumbleOnceAndSecondContactCanEndEpisode() throws {
        var model = try empty()
        model.entities = [MetroRunnerEntity(kind: .bush, x: 0, y: 0, z: 8, width: 12, depth: 12, height: 5)]
        for _ in 0..<15 { _ = try model.step(0) }
        #expect(!model.dead && model.stumble > 0 && model.entities[0].contacted)
        model.invulnerable = 0
        model.entities.append(MetroRunnerEntity(kind: .powerbox, x: 0, y: 0, z: model.motor.z + 3,
                                                width: 18, depth: 4, height: 8))
        #expect(try model.step(0).terminated)
    }

    @Test
    func recoveredDurationsAndMysteryAccounting() throws {
        var model = try empty()
        #expect(model.catalog.rules.duration(.magnet, level: 0) == 10)
        #expect(model.catalog.rules.duration(.jetpack, level: 0) == 8)
        #expect(model.catalog.rules.duration(.magnet, level: 6) == 30)
        #expect(model.catalog.rules.duration(.jetpack, level: 6) == 25)
        #expect(throws: GymnazoError.self) { try empty(level: 7) }
        #expect(throws: GymnazoError.self) { try empty(level: -1) }
        model.entities = [MetroRunnerEntity(kind: .mystery, x: 0, y: 5, z: model.motor.z)]
        model.collect(0)
        #expect(model.coins == 0 && model.mysteryBoxes == 1)
        #expect(model.info["original_game_validated"] == .bool(false))
    }

    @Test
    func sneakersCustomAscentSurvivesPowerExpiryAndRecoversStumble() throws {
        var model = try empty()
        model.stumble = 4
        model.entities = [MetroRunnerEntity(kind: .sneakers, x: 0, y: 5, z: 0)]
        model.collect(0)
        #expect(model.stumble == 0)
        _ = try model.step(MetroRunnerAction.jump.rawValue)
        let duration = try #require(model.motor.customJump).duration
        #expect(abs(duration - sqrt(80.0 / 200)) < 1e-12)
        model.powers[.sneakers] = 0.001
        for _ in 0..<6 { _ = try model.step(0) }
        #expect(model.motor.customJump != nil && model.motor.y > 20)
        for _ in 0..<200 { _ = try model.step(0) }
        #expect(model.motor.grounded && model.motor.customJump == nil)
    }

    @Test
    func pogoHasExclusiveRowsAirborneRewardAndFastDive() throws {
        var model = try empty()
        model.startPogo()
        let coins = model.entities.filter { $0.kind == .coin }
        #expect(coins.count == 45 && Set(coins.map(\.x)) == [-20, 0, 20])
        #expect(coins.map(\.z).min() == 58 && coins.map(\.z).max()! < 300)
        #expect(coins.allSatisfy { $0.inAir })
        let reward = try #require(model.entities.first { $0.kind.isPower })
        #expect(reward.z == 315 && reward.y == 162)
        #expect(!model.motor.canJump)
        guard case .floatTensor(let player) = try model.observation()["player"] else { return }
        #expect(player.elements[33] == 1 && player.elements[34] == 1)
        for _ in 0..<100 { _ = try model.step(0) }
        #expect(model.motor.y > 100 && model.motor.y <= 160)
        _ = try model.step(MetroRunnerAction.down.rawValue)
        #expect(model.motor.diving && model.motor.customJump == nil && model.motor.vy < -250)
    }

    @Test
    func jetpackSmoothstepCapturedSpeedAndAdjacentCoinPath() throws {
        var model = try empty()
        model.entities = [MetroRunnerEntity(kind: .jetpack, x: 0, y: 5, z: 0)]
        model.collect(0)
        let coins = model.entities.filter { $0.kind == .coin }.sorted { $0.z < $1.z }
        #expect(coins.first?.z == 530)
        #expect(coins.allSatisfy { $0.y == 101 })
        for (a, b) in zip(coins, coins.dropFirst()) {
            #expect(abs(b.x - a.x) <= 10)
            #expect(abs(b.z - a.z - 30) < 1e-8)
        }
        for _ in 0..<15 { _ = try model.step(0) }
        #expect(abs(model.motor.y - 95 * pow(1.0 / 6, 2) * (3 - 2.0 / 6)) < 1e-10)
        #expect(model.motor.vy > 0 && model.speed == 220)
        for _ in 0..<90 { _ = try model.step(MetroRunnerAction.jump.rawValue) }
        #expect(model.motor.y == 95 && model.speed == 220 && model.jumps == 0)
    }

    @Test
    func flyoverUsesAnchorPreserveBoundaryAndStrictLandingEndpoint() throws {
        var model = try empty()
        model.flightEndZ = 1760
        #expect(!model.flyoverSuppresses(MetroRunnerEntity(kind: .coin, x: 0, y: 6, z: 599, source: "chunk")))
        #expect(model.flyoverSuppresses(MetroRunnerEntity(kind: .coin, x: 0, y: 6, z: 600, source: "chunk")))
        #expect(!model.flyoverSuppresses(MetroRunnerEntity(kind: .coin, x: 0, y: 6, z: 1760, source: "chunk")))
        #expect(model.flyoverSuppresses(MetroRunnerEntity(kind: .coin, x: 0, y: 6, z: 10, source: "chunk", disableOnFlyover: true)))
        #expect(!model.flyoverSuppresses(MetroRunnerEntity(kind: .coin, x: 0, y: 6, z: 700, source: "chunk",
                                                          spawnOrigin: SIMD3(0, 6, 599))))
    }

    @Test
    func observationVersionFiveDistinguishesFastTrainsAndBeamBounds() throws {
        var model = try empty()
        model.motor.elapsed = 179
        model.entities = [MetroRunnerEntity(kind: .movingTrain, x: 0, y: 1.2, z: 100, velocity: -330)]
        guard case .floatTensor(let a) = try model.observation()["entities"] else { return }
        model.entities[0].velocity = -440
        guard case .floatTensor(let b) = try model.observation()["entities"] else { return }
        #expect(abs(a.elements[27] - Float(-550.0 / 1320)) < 1e-6)
        #expect(abs(b.elements[27] - Float(-660.0 / 1320)) < 1e-6)
        model.entities = [MetroRunnerEntity(kind: .roll, x: 0, y: 0, z: 10)]
        guard case .floatTensor(let beam) = try model.observation()["entities"] else { return }
        #expect(beam.elements[25] == Float(5.0 / 200) && beam.elements[26] == Float(24.0 / 200))
        #expect(MetroRunnerObservation.flatCount == 2089)
        #expect(MetroRunnerObservation.version == 5 && MetroRunnerObservation.playerFeatures == 41)
        guard case .floatTensor(let player) = try model.observation()["player"] else {
            Issue.record("Missing player observation")
            return
        }
        #expect(player.shape == [41] && player.elements[29] == 1)
        #expect(try MetroRunnerObservation.space().containsValue(model.observation()))
    }
}
