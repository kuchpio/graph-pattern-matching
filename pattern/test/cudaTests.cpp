#ifdef CUDA_ENABLED

#include "cuda_subgraph_matcher.h"
#include "gtest/gtest.h"
#include "utils.h"

namespace pattern
{
TEST(CudaSubgraphIsomorphism, small_subgraph) {
    // Create the larger graph G
    std::size_t graphSize = 5;
    std::size_t subgraphSize = 4;

    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(subgraphSize);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(0, 2);
    G.add_edge(0, 3);
    G.add_edge(0, 4);

    G.add_edge(1, 0);
    G.add_edge(1, 2);
    G.add_edge(1, 3);
    G.add_edge(1, 4);

    G.add_edge(2, 1);
    G.add_edge(2, 3);
    G.add_edge(2, 4);
    G.add_edge(2, 0);

    G.add_edge(3, 0);
    G.add_edge(3, 1);
    G.add_edge(3, 2);
    G.add_edge(3, 4);

    G.add_edge(4, 0);
    G.add_edge(4, 1);
    G.add_edge(4, 2);
    G.add_edge(4, 3);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    // Check for subgraph isomorphism
    auto matcher = CudaSubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    EXPECT_TRUE(utils::MatchingChecker::checkSubgraphMatching(G, Q, matching.value()));
}

TEST(CudaSubgraphIsomorphism, randomBigSearchGraph) {
    srand(SEED);

    std::size_t bigGraphSize = 600;
    std::size_t smallGraphSize = 20;

    auto G = utils::GraphFactory::random_connected_graph(bigGraphSize, 0.04f);
    auto Q = utils::GraphFactory::random_connected_graph(smallGraphSize, 0.8f);

    auto matcher = CudaSubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_FALSE(matching.has_value());
}
} // namespace pattern
#endif
