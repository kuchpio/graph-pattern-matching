#include "core.h"
#include "minor_matchers.h"

#include "utils.h"

#include "gtest/gtest.h"
namespace pattern
{

TEST(minorMiner, SmallNotMinor) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G (a linear chain with no cycles)
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller graph Q (a cycle, which cannot be formed by contractions in G)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = MinerMinorMatcher();

    // Check for minor relationship - expecting false because Q is a cycle but G is a chain
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(minorMiner, SmallHasMinor) {
    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G (Qubic graph Q^3)
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    // Define edges for the smaller graph Q (wheel W^4)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = MinerMinorMatcher();

    // Check for minor relationship - expecting true because Q can be derived from G
    EXPECT_TRUE(matcher.match(G, Q).has_value());
}

TEST(TopologicalMinorIsomorphism, HasMinorNotTopological) {
    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G (Qubic graph Q^3)
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    // Define edges for the smaller graph Q (wheel W^4)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = TopologicalMinorExactMatcher();

    // Check for minor relationship - expecting true because Q can be derived from G
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}
TEST(TopologicalMinorIsomorphism, HasTopologicalMinor) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);
    G.add_edge(5, 0);
    G.add_edge(1, 4);
    G.add_edge(4, 1);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(1, 3);
    Q.add_edge(3, 1);

    auto matcher = TopologicalMinorExactMatcher();

    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    auto correctMatching = std::vector<vertex>{1, 1, 1, 2, 3, 0};
    EXPECT_TRUE(utils::MatchingChecker::checkMinorMatching(G, Q, matching.value()));
}

TEST(InducedMinorIsomorphism, SmallNotInducedMinor) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = InducedMinorExactMatcher();

    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(InducedMinorIsomorphism, SmallHasInducedMinor) {
    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = InducedMinorExactMatcher();

    EXPECT_TRUE(matcher.match(G, Q).has_value());
}

TEST(InducedTopologicalMinorIsomorphism, SmallNotInducedTopologicalMinor) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = InducedTopologicalMinorExactMatcher();

    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(InducedTopologicalMinorIsomorphism, SmallHasInducedButNotTopologicalMinor) {
    std::size_t graph_size = 8;
    std::size_t subgraph_size = 5;

    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 0);
    G.add_edge(0, 4);
    G.add_edge(1, 5);
    G.add_edge(2, 6);
    G.add_edge(3, 7);
    G.add_edge(4, 5);
    G.add_edge(5, 6);
    G.add_edge(6, 7);
    G.add_edge(7, 4);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(0, 4);
    Q.add_edge(1, 4);
    Q.add_edge(2, 4);
    Q.add_edge(3, 4);

    auto matcher = InducedTopologicalMinorExactMatcher();

    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(MinorMiner, random_120_vertex) {
    srand(SEED);

    auto G = utils::GraphFactory::random_connected_graph(120, 0.3);
    auto minor = utils::GraphFactory::random_minor(G, 45);

    auto matcher = MinerMinorMatcher();

    auto matching = matcher.match(G, minor);

    EXPECT_TRUE(matching.has_value());
}

TEST(topologicalMinor, random_100_vertex) {
    srand(SEED);

    auto topologicalMinor = utils::GraphFactory::random_connected_graph(70);
    auto G = utils::GraphFactory::random_edge_subdivisions(topologicalMinor, 30);

    auto matcher = TopologicalMinorExactMatcher(true);

    auto matching = matcher.match(G, topologicalMinor);
    EXPECT_TRUE(matching.has_value());
}

TEST(indcuedTopologicalMinor, random_130_vertex) {
    srand(SEED);

    auto topologicalMinor = utils::GraphFactory::random_connected_graph(100);
    auto G = utils::GraphFactory::random_edge_subdivisions(topologicalMinor, 30);

    auto matcher = InducedTopologicalMinorExactMatcher(true);

    auto matching = matcher.match(G, topologicalMinor);
    EXPECT_TRUE(matching.has_value());
}

TEST(inducedMinor, random_40_vertex) {
    srand(SEED);

    auto G = utils::GraphFactory::random_connected_graph(60, 0.9);
    auto inducedMinor = utils::GraphFactory::random_induced_minor(G, 8);

    auto matcher = InducedMinorExactMatcher(true);

    auto matching = matcher.match(G, inducedMinor);
    EXPECT_TRUE(matching.has_value());
}

} // namespace pattern