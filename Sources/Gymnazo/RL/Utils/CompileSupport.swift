func supportsCompilation(_ extractor: any FeaturesExtractor) -> Bool {
    extractor is FlattenExtractor || extractor is NatureCNN
}
