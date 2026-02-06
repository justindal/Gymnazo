import Testing
import MLX
@testable import Gymnazo

@Suite("Graph space sampling and membership")
struct GraphSpaceTests {
    @Test
    func testSampleContains() async throws {
        let nodeSpace = Box(low: -1.0, high: 1.0, shape: [3], dtype: .float32)
        let edgeSpace = Box(low: 0.0, high: 1.0, shape: [2], dtype: .float32)
        let space = Graph(nodeSpace: nodeSpace, edgeSpace: edgeSpace, maxNodes: 4, maxEdges: 6)

        var key = MLX.key(2025)
        for _ in 0..<10 {
            let (k, _) = MLX.split(key: key)
            key = k
            let flat = space.sample(key: key)
            #expect(space.contains(flat))
            let sample = space.sampleGraph(key: key, mask: nil, probability: nil)
            #expect(sample.nodes.shape == [4, 3])
            #expect(sample.edges.shape == [6, 2])
            #expect(sample.edgeLinks.shape == [6, 2])
            #expect(sample.nodeMask.shape == [4])
            #expect(sample.edgeMask.shape == [6])
        }
    }

    @Test
    func testPaddedEdgeLinksAreNegativeOne() async throws {
        let nodeSpace = MultiBinary(n: 2)
        let edgeSpace = MultiBinary(n: 1)
        let space = Graph(nodeSpace: nodeSpace, edgeSpace: edgeSpace, maxNodes: 3, maxEdges: 5)
        let sample = space.sampleGraph(key: MLX.key(1), mask: nil, probability: nil)

        let edgeMask = sample.edgeMask.asType(.bool).asArray(Bool.self)
        var edgePrefix = 0
        while edgePrefix < edgeMask.count && edgeMask[edgePrefix] {
            edgePrefix += 1
        }

        let links = sample.edgeLinks.asType(.int32).reshaped([10]).asArray(Int32.self)
        for e in edgePrefix..<5 {
            #expect(links[e * 2] == -1)
            #expect(links[e * 2 + 1] == -1)
        }
    }
}
