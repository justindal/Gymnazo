import MLX
import Testing

@testable import Gymnazo

@Suite("Metro Run native integration", .serialized)
struct MetroRunnerTests {
    private func empty(frameSkip: Int = 1, seconds: Double = 300) throws -> MetroRunnerModel {
        var model = try MetroRunnerModel(config: MetroRunnerConfig(frameSkip: frameSkip, maxEpisodeSeconds: seconds))
        _ = try model.reset(seed: 11, options: nil)
        model.entities = []
        model.nextChunk = 1e9
        return model
    }

    private func advance(_ model: inout MetroRunnerModel, ticks: Int, action: Int = 0) throws {
        for index in 0..<ticks {
            let result = try model.step(index == 0 ? action : 0)
            if result.terminated || result.truncated { break }
        }
    }

    @Test
    func catalogPreservesAllLayoutsAndNormalReachability() throws {
        let catalog = try MetroRunnerCatalog.bundled.get()
        #expect(catalog.chunks.count == 49)
        #expect(catalog.sections.count == 49)
        #expect(catalog.entries.count == 39)
        func chunks(_ section: String) -> Set<String> {
            guard let value = catalog.sections[section] else { return [] }
            return value.sections.reduce(Set(value.chunks)) { $0.union(chunks($1)) }
        }
        let normal = catalog.entries.reduce(Set<String>()) { $0.union(chunks($1.section)) }
        let all = catalog.sections.keys.reduce(Set<String>()) { $0.union(chunks($1)) }
        #expect(normal.count == 42)
        #expect(all == Set(catalog.chunks.keys))
        var route = try MetroRunnerRoute(catalog: catalog, seed: 7, section: nil)
        for name in catalog.chunks.keys.sorted() {
            let (entities, length) = try route.instantiate(name, origin: 0, speed: 110)
            #expect(length >= 180)
            #expect(entities.allSatisfy { ($0.x + $0.y + $0.z + $0.width + $0.depth + $0.height).isFinite })
        }
    }

    @Test
    func schedulerPreservesConnectionsDistanceBoundsAndRepeatSpacing() throws {
        let catalog = try MetroRunnerCatalog.bundled.get()
        var route = try MetroRunnerRoute(catalog: catalog, seed: 7, section: nil)
        #expect(try route.next(at: 0) == "Chunk_Intro")
        route.placed(at: 180)
        #expect(["Chunk_Start1", "Chunk_Start2"].contains(try route.next(at: 180)))
        for (section, prefix) in [("Section_1_Track", "Chunk_1_Track"), ("Section_2_Tracks", "Chunk_2_Tracks")] {
            let first = try route.expand(section)
            let second = try route.expand(section)
            #expect(first.count == 3)
            #expect(first.first == prefix + "_Start")
            #expect(first.last == prefix + "_End")
            #expect(first[1] != second[1])
        }
        func names(_ distance: Double) -> Set<String> { Set(route.eligible(at: distance).map { catalog.entries[$0].section }) }
        #expect(!names(4000).contains("Section_Train_Tops_Moving"))
        #expect(names(4001).contains("Section_Train_Tops_Moving"))
        #expect(!names(12000).contains("Section_Short_Train_Tops_Moving"))
        #expect(names(12001).contains("Section_Short_Train_Tops_Moving"))
        route.lastPlacement["Section_Tunnel"] = 10000
        #expect(!names(11899).contains("Section_Tunnel"))
        #expect(names(11900).contains("Section_Tunnel"))
    }

    @Test
    func exclusiveBarrierGroupsAndRecoveredTrainGeometry() throws {
        var route = try MetroRunnerRoute(catalog: MetroRunnerCatalog.bundled.get(), seed: 7, section: nil)
        let (items, _) = try route.instantiate("Chunk_S-B-S-B", origin: 0, speed: 110)
        let barriers = items.filter { $0.kind == .jump || $0.kind == .roll }
        #expect(barriers.count == 2)
        #expect(barriers.map { Int($0.z.rounded()) }.sorted() == [150, 585])
        let (trains, _) = try route.instantiate("Chunk_S-S", origin: 0, speed: 110)
        let staticTrains = trains.filter { $0.kind == .train }
        #expect(!staticTrains.isEmpty)
        for train in staticTrains {
            #expect(abs(train.width - 17) < 1e-6)
            #expect(abs(train.y + train.height - 29.2) < 1e-5)
            #expect([55.5, 118, 176, 296].contains((train.depth * 10).rounded() / 10))
        }
    }

    @Test
    func motorMatchesPythonReferenceTrace() {
        var motor = MetroRunnerMotor()
        let reference: [(Int, Double, Double, Double, Double)] = [
            (0, 0.9382716049382716, 0, 0, 1.8333333333333333),
            (3, 0.8765604524449963, 0, 0, 7.334357541899442),
            (8, 1, 1.4351564294443044, 86.10938576665826, 16.50614525139665),
            (16, 1, 10.916407864998742, 59.44271909999162, 31.189882060831792),
            (22, 1, 13.481078901109154, -93.33333333333333, 42.209854127870905),
            (39, 1, 0, 0, 73.46648044692739),
        ]
        for tick in 0..<40 {
            let command: MetroRunnerAction = [0: .left, 3: .right, 8: .jump, 22: .down][tick] ?? .none
            motor.step(command)
            if let expected = reference.first(where: { $0.0 == tick }) {
                #expect(abs(motor.x - expected.1) < 1e-10)
                #expect(abs(motor.y - expected.2) < 1e-10)
                #expect(abs(motor.vy - expected.3) < 1e-10)
                #expect(abs(motor.z - expected.4) < 1e-10)
            }
        }
        #expect(motor.grounded)
        motor.elapsed = 179
        motor.step(.left)
        #expect(abs(motor.laneDuration - 0.135) < 1e-10)
    }

    @Test
    func frameSkipAppliesOneInputAndTimeLimitStopsOnExactTick() throws {
        var model = try empty(frameSkip: 40, seconds: 0.1)
        let result = try model.step(MetroRunnerAction.left.rawValue)
        #expect(model.motor.lane == 0)
        #expect(model.motor.y == 0)
        #expect(model.physicsSteps == 6)
        #expect(result.truncated && !result.terminated)
        #expect(result.info["action_physics_steps"] == .discrete(6))
        #expect(throws: GymnazoError.self) { try model.step(0) }
        var impulse = try empty(frameSkip: 40)
        impulse.motor.lane = 0
        impulse.motor.x = 0
        impulse.motor.laneFrom = 0
        impulse.motor.laneTo = 0
        _ = try impulse.step(MetroRunnerAction.right.rawValue)
        #expect(impulse.motor.lane == 1)
        #expect(impulse.motor.x == 1)
        var jump = try empty(frameSkip: 90)
        _ = try jump.step(MetroRunnerAction.jump.rawValue)
        #expect(jump.motor.grounded && jump.jumps == 1)
    }

    @Test
    func actionsExcludeHoverboardsAndRejectTheFormerBoardIndex() throws {
        #expect(MetroRunnerAction.allCases.map(\.rawValue) == [0, 1, 2, 3, 4])
        var legacy = try MetroRunner()
        var typed = try MetroRunnerValueEnvironment()
        #expect((legacy.actionSpace as? Discrete)?.n == 5)
        #expect(typed.valueActionSpace.n == 5)
        _ = try legacy.reset(seed: 7)
        _ = try typed.resetValue(seed: 7)
        let before = legacy.currentSnapshot
        #expect(throws: GymnazoError.self) { try legacy.step(MLXArray(Int32(5))) }
        #expect(throws: GymnazoError.self) { try typed.stepValue(5) }
        #expect(legacy.currentSnapshot == before)
        #expect(typed.currentSnapshot == before)
        for action in 0..<5 {
            #expect(typed.valueActionSpace.containsValue(action))
        }
        #expect(!typed.valueActionSpace.containsValue(5))
    }

    @Test
    func frontalCrashTerminatesWithoutClearingNearbyHazards() throws {
        var model = try empty(frameSkip: 4)
        model.entities = [
            MetroRunnerEntity(kind: .train, x: 0, y: 0, z: 1, width: 16, depth: 180, height: 30),
            MetroRunnerEntity(kind: .jump, x: 0, y: 0, z: 100),
        ]
        let result = try model.step(MetroRunnerAction.none.rawValue)
        #expect(result.terminated && !result.truncated)
        #expect(model.snapshot.dead && model.physicsSteps == 1)
        #expect(model.entities.allSatisfy { $0.alive })
        #expect(model.deathReason == "Train collision")
        #expect(throws: GymnazoError.self) { try model.step(0) }
    }

    @Test
    func barriersRequireMatchingActionAndTrainCrashTerminatesEarly() throws {
        var standing = try empty(frameSkip: 20)
        standing.entities = [MetroRunnerEntity(kind: .roll, x: 0, y: 0, z: 8)]
        let crash = try standing.step(0)
        #expect(crash.terminated && !crash.truncated)
        #expect(standing.physicsSteps < 20)
        #expect(standing.deathReason == "Roll under the overhead barrier")
        var rolling = try empty()
        rolling.entities = [MetroRunnerEntity(kind: .roll, x: 0, y: 0, z: 8)]
        try advance(&rolling, ticks: 20, action: MetroRunnerAction.down.rawValue)
        #expect(!rolling.dead)
        var jumping = try empty()
        jumping.entities = [MetroRunnerEntity(kind: .jump, x: 0, y: 0, z: 45, height: 12)]
        try advance(&jumping, ticks: 45, action: MetroRunnerAction.jump.rawValue)
        #expect(!jumping.dead)
        var train = try empty()
        train.entities = [MetroRunnerEntity(kind: .train, x: 0, y: 0, z: 8, width: 16, depth: 180, height: 30)]
        try advance(&train, ticks: 20)
        #expect(train.dead)
        var boundary = try empty(frameSkip: 4, seconds: 0.01)
        boundary.entities = [MetroRunnerEntity(kind: .train, x: 0, y: 0, z: 1, height: 30)]
        let lastTick = try boundary.step(0)
        #expect(lastTick.terminated && !lastTick.truncated)
        #expect(lastTick.info["action_physics_steps"] == .discrete(1))
    }

    @Test
    func rampRoofAndFallMatchPythonReference() throws {
        var model = try empty()
        model.entities = [MetroRunnerEntity(kind: .ramp, x: 0, y: 0, z: 10, width: 16, depth: 30, height: 30),
                          MetroRunnerEntity(kind: .train, x: 0, y: 0, z: 40, width: 16, depth: 60, height: 30)]
        for tick in 0..<65 {
            _ = try model.step(0)
            #expect(!model.dead)
            if tick == 9 { #expect(abs(model.motor.y - 8.341014897579143) < 1e-10) }
            if tick == 19 { #expect(abs(model.motor.y - 26.699099937926768) < 1e-10) }
            if tick == 39 { #expect(model.motor.y == 30 && model.motor.grounded) }
        }
        #expect(abs(model.motor.y - 27.5) < 1e-10)
        #expect(!model.motor.grounded)

        var recovered = try empty()
        var route = try #require(recovered.route)
        let (items, _) = try route.instantiate("Chunk_Ramp_1", origin: 0, speed: 110)
        recovered.entities = items.filter { $0.kind == .train || $0.kind == .ramp }
        let ramp = try #require(recovered.entities.first { $0.kind == .ramp })
        let lane = Int(ramp.x / 20 + 1)
        recovered.motor.lane = lane
        recovered.motor.x = Double(lane)
        recovered.motor.laneFrom = Double(lane)
        recovered.motor.laneTo = Double(lane)
        for _ in 0..<85 {
            _ = try recovered.step(0)
            #expect(!recovered.dead)
            if ramp.z + 62 <= recovered.motor.z, recovered.motor.z <= 170 {
                #expect(abs(recovered.motor.y - 29.2) < 0.1)
            }
        }
    }

    @Test
    func movingTrainMappingRetainsRelativeContactDuration() throws {
        var model = try empty()
        var route = try #require(model.route)
        let (items, _) = try route.instantiate("Chunk_Train_Tops_Moving", origin: 0, speed: 110)
        let train = try #require(items.first { $0.kind == .movingTrain })
        model.entities = [train]
        model.updateMovingTrains(at: 20)
        let firstZ = model.entities[0].z
        model.updateMovingTrains(at: 30)
        let second = model.entities[0]
        #expect(abs(second.z - firstZ + 10 * train.motionFactor) < 1e-10)
        #expect(second.velocity == -train.motionFactor * model.motor.speed)
        #expect(abs(second.depth / (model.motor.speed - second.velocity) - second.depth / (1 + train.motionFactor) / model.motor.speed) < 1e-10)
    }

    @Test
    func observationIncludesOverlappingSupportAndReservesHazardRows() throws {
        var model = try empty()
        model.motor.z = 50
        model.entities = (0..<100).map { MetroRunnerEntity(kind: .coin, x: 0, y: 5, z: 51 + Double($0)) }
        model.entities.append(MetroRunnerEntity(kind: .train, x: 0, y: 1.2, z: 0, depth: 176, height: 28))
        let value = try model.observation()
        #expect(try MetroRunnerObservation.space().containsValue(value))
        guard case .floatTensor(let rows) = value["entities"], case .boolTensor(let mask) = value["entity_mask"] else {
            Issue.record("Missing observation components")
            return
        }
        #expect(rows.elements[0] == 1)
        #expect(rows.elements[23] < 0 && rows.elements[24] > 0)
        #expect(abs(rows.elements[28] - Float(29.2 / 150)) < 1e-6)
        #expect(mask.elements.allSatisfy { $0 })
        model.entities = []
        let emptyValue = try model.observation()
        guard case .boolTensor(let emptyMask) = emptyValue["entity_mask"] else { return }
        #expect(!emptyMask.elements.contains(true))
        #expect(try MetroRunnerObservation.flatten(emptyValue).elements.allSatisfy(\.isFinite))
    }

    @Test
    func pickupsRewardAndGameplayRandomnessLeaveRouteUnchanged() throws {
        var first = try empty()
        first.entities = [MetroRunnerEntity(kind: .coin, x: 20, y: 6, z: 50)]
        first.powers[.magnet] = 15
        let reward = try first.step(0)
        #expect(first.coins == 1)
        #expect(abs(reward.reward - (first.motor.z * 0.01 + 0.05)) < 1e-10)
        var a = try empty(), b = try empty()
        a.entities = [MetroRunnerEntity(kind: .jetpack, x: 0, y: 6, z: 0), MetroRunnerEntity(kind: .mystery, x: 0, y: 6, z: 0)]
        a.collect(0); a.collect(1)
        for _ in 0..<120 { _ = try a.step(0); _ = try b.step(0) }
        #expect(a.motor.y > 60)
        var routeA = try #require(a.route), routeB = try #require(b.route)
        var origin = 2400.0
        for _ in 0..<20 {
            let nameA = try routeA.next(at: origin), nameB = try routeB.next(at: origin)
            #expect(nameA == nameB)
            let (itemsA, length) = try routeA.instantiate(nameA, origin: origin, speed: 110)
            let (itemsB, _) = try routeB.instantiate(nameB, origin: origin, speed: 110)
            #expect(itemsA == itemsB)
            origin += length
            routeA.placed(at: origin); routeB.placed(at: origin)
        }
    }

    @Test
    func seedsReplayAndLegacyTypedTracesAgree() throws {
        var legacy = try MetroRunner()
        var typed = try MetroRunnerValueEnvironment()
        let reset = try legacy.reset(seed: 42)
        let valueReset = try typed.resetValue(seed: 42)
        #expect(reset.obs.asArray(Float.self) == (try MetroRunnerObservation.flatten(valueReset.observation).elements))
        #expect(legacy.observationSpace.contains(reset.obs))
        let snapshot = legacy.currentSnapshot
        for action in [1, 0, 3, 0, 4, 2, 0] {
            let step = try legacy.step(MLXArray(Int32(action)))
            let valueStep = try typed.stepValue(action)
            #expect(step.obs.asArray(Float.self) == (try MetroRunnerObservation.flatten(valueStep.observation).elements))
            #expect(step.reward == valueStep.reward)
            #expect(step.terminated == valueStep.terminated && step.truncated == valueStep.truncated)
            if step.terminated || step.truncated { break }
        }
        _ = try legacy.reset(seed: 42)
        #expect(legacy.currentSnapshot == snapshot)
        let fresh = try legacy.reset()
        #expect(fresh.info["route_seed"]?.string != reset.info["route_seed"]?.string)
        let replay = try legacy.reset(options: ["route_seed": try #require(reset.info["route_seed"]?.string)])
        #expect(replay.obs.asArray(Float.self) == reset.obs.asArray(Float.self))
    }

    @Test
    func validatesLifecycleConfigurationActionsAndOptions() throws {
        #expect(throws: GymnazoError.self) { try MetroRunner(config: MetroRunnerConfig(frameSkip: 0)) }
        #expect(throws: GymnazoError.self) { try MetroRunner(config: MetroRunnerConfig(maxEpisodeSeconds: .infinity)) }
        #expect(throws: GymnazoError.self) { try MetroRunner(config: MetroRunnerConfig(section: "missing")) }
        #expect(throws: GymnazoError.self) { try MetroRunner(config: MetroRunnerConfig(reward: MetroRunnerReward(coinWeight: .nan))) }
        var env = try MetroRunner()
        #expect(throws: GymnazoError.self) { try env.step(MLXArray(Int32(0))) }
        _ = try env.reset(seed: 7)
        for action in [MLXArray(Int32(-1)), MLXArray(Int32(5)), MLXArray(Int32(6)), MLXArray(Float(1.5)), MLXArray([Int32(1), 2])] {
            #expect(throws: GymnazoError.self) { try env.step(action) }
        }
        #expect(throws: GymnazoError.self) { try env.reset(options: ["unknown": 1]) }
        #expect(throws: GymnazoError.self) { try env.reset(options: ["curriculum": "missing"]) }
        env.close()
        #expect(throws: GymnazoError.self) { try env.reset(seed: 7) }
        #expect(throws: GymnazoError.self) { try env.step(MLXArray(Int32(0))) }
    }

    @Test
    func curriculaFilterGeometryAndLongSeededRolloutsStayBounded() throws {
        for curriculum in MetroRunnerCurriculum.allCases {
            var model = try MetroRunnerModel(config: MetroRunnerConfig(curriculum: curriculum))
            _ = try model.reset(seed: 9, options: nil)
            if curriculum == .barriers {
                #expect(model.entities.allSatisfy { [.jump, .roll, .coin].contains($0.kind) && $0.y < 15 })
            }
            if curriculum == .static { #expect(!model.entities.contains { $0.kind == .movingTrain }) }
            if curriculum != .full { #expect(!model.entities.contains { $0.kind.isPower }) }
        }
        var first = try MetroRunnerModel(config: MetroRunnerConfig())
        var second = first
        _ = try first.reset(seed: 22, options: nil)
        _ = try second.reset(seed: 22, options: nil)
        first.invulnerable = 10000; second.invulnerable = 10000
        var maximumEntities = 0
        for step in 0..<3750 {
            let action = step % 23 == 0 ? 3 : 0
            let a = try first.step(action), b = try second.step(action)
            #expect(a.observation == b.observation)
            #expect(!a.terminated && !a.truncated)
            maximumEntities = max(maximumEntities, first.entities.count)
        }
        #expect(first.motor.z > 30000)
        #expect(maximumEntities < 1500)
        #expect(first.snapshot == second.snapshot)
    }

    @Test
    func registryAndParallelValueVectorsSupportMetroRun() async throws {
        var env = try await Gymnazo.make("MetroRunner-v0")
        #expect(env.spec?.id == "MetroRunner-v0")
        #expect(env.spec?.metadata.renderModes == [.human, .ansi])
        let reset = try env.reset(seed: 7)
        #expect(env.observationSpace.shape == [2089])
        #expect(env.observationSpace.contains(reset.obs))
        _ = try env.step(MLXArray(Int32(0)))
        let value = try await Gymnazo.makeValue("MetroRunner-v0")
        #expect(value.observationSpace.flatDimension == 2089)
        let vector = try await Gymnazo.makeValueVec("MetroRunner-v0", numEnvs: 2, mode: .parallel,
            options: ["config": MetroRunnerConfig(maxEpisodeSeconds: 0.05)])
        _ = try await vector.reset(seed: 7)
        let result = try await vector.step([.discrete(1), .discrete(2)])
        #expect(result.truncations == [true, true])
        try await vector.close()
    }

    @Test
    @MainActor
    func ppoUpdatesParametersWithFlattenedNativeEnvironments() async throws {
        let model = try PPO(
            observationSpace: Box(low: -1, high: 1, shape: [MetroRunnerObservation.flatCount]),
            actionSpace: Discrete(n: 5),
            policyConfig: PPOPolicyConfig(netArch: .shared([16]), featuresExtractor: .flatten),
            config: PPOConfig(nSteps: 4, batchSize: 4, nEpochs: 1), seed: 7)
        let before = await model.policyParameterValues()
        try await model.learn(environments: [
            try MetroRunnerValueEnvironment(config: MetroRunnerConfig(maxEpisodeSeconds: 0.1)).flattened(),
            try MetroRunnerValueEnvironment(config: MetroRunnerConfig(maxEpisodeSeconds: 0.1)).flattened(),
        ], totalTimesteps: 16)
        #expect(await model.timesteps == 16)
        #expect(await model.policyParameterValues() != before)
    }
}
