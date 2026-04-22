import Testing
import MLX
@testable import Gymnazo

@Suite("Robustness error surfaces")
struct RobustnessTests {
    @Test
    func testNatureCnnExtractorOnDictThrowsTypedError() throws {
        let observationSpace = Dict([
            "state": Box(low: -1.0, high: 1.0, shape: [4], dtype: .float32)
        ])

        do {
            _ = try FeaturesExtractorConfig.natureCNN().make(
                observationSpace: observationSpace,
                normalizeImages: true
            )
            Issue.record("Expected natureCNN extractor creation to throw on Dict spaces")
        } catch let error as GymnazoError {
            #expect(
                error
                    == .invalidFeatureExtractorConfiguration(
                        config: "natureCNN",
                        observationSpace: "Dict"
                    )
            )
        } catch {
            Issue.record("Expected GymnazoError, got \(error)")
        }
    }

    @Test
    func testCombinedExtractorOnBoxThrowsTypedError() throws {
        let observationSpace = Box(low: -1.0, high: 1.0, shape: [4], dtype: .float32)

        do {
            _ = try FeaturesExtractorConfig.combined().make(
                observationSpace: observationSpace,
                normalizeImages: true
            )
            Issue.record("Expected combined extractor creation to throw on Box spaces")
        } catch let error as GymnazoError {
            #expect(
                error
                    == .invalidFeatureExtractorConfiguration(
                        config: "combined",
                        observationSpace: "Box"
                    )
            )
        } catch {
            Issue.record("Expected GymnazoError, got \(error)")
        }
    }

    @Test
    func testDistributionFactoryUnsupportedSpaceThrowsTypedError() throws {
        let actionSpace = TextSpace(minLength: 0, maxLength: 8)

        do {
            _ = try DistributionFactory.makeProbaDistribution(actionSpace: actionSpace)
            Issue.record("Expected distribution factory to throw for TextSpace")
        } catch let error as GymnazoError {
            #expect(error == .unsupportedDistributionSpace(spaceType: "TextSpace"))
        } catch {
            Issue.record("Expected GymnazoError, got \(error)")
        }
    }
}
