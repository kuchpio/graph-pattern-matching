#include "core.h"
#include "miner_minor_matcher.hpp"
#include "native_minor_matcher.h"
#include "topological_minor_matcher.h"
#include "gtest/gtest.h"
namespace pattern
{
TEST(NativeMinorIsomorphism, SmallNotMinor) {
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

    auto matcher = NativeMinorMatcher();

    // Check for minor relationship - expecting false because Q is a cycle but G is a chain
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

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

TEST(NativeMinorIsomorphism, SmallHasMinor) {
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

    auto matcher = NativeMinorMatcher();

    // Check for minor relationship - expecting true because Q can be derived from G
    EXPECT_TRUE(matcher.match(G, Q).has_value());
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

    auto matcher = TopologicalMinorMatcher();

    // Check for minor relationship - expecting true because Q can be derived from G
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}
} // namespace pattern