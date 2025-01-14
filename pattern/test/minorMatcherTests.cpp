#include "core.h"
#include "miner_minor_matcher.hpp"
#include "native_minor_matcher.h"
#include "topological_minor_matcher.h"
#include "topological_minor_heuristic_solver.h"
#include "induced_minor_heuristic.h"
#include "topological_induced_minor_heuristic_solver.h"
#include "utils.h"

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

    auto matcher = TopologicalMinorHeuristicSolver();

    // Check for minor relationship - expecting true because Q can be derived from G
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}
TEST(TopologicalMinorIsomorphism, HasTopologicalMinor) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    // G forms a "hexagon" with an additional edge
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);
    G.add_edge(5, 0);
    G.add_edge(1, 4);
    G.add_edge(4, 1); // Extra edge to make Q a topological minor

    // Define edges for the smaller graph Q (square with a diagonal)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);
    Q.add_edge(1, 3);
    Q.add_edge(3, 1); // Diagonal edge

    auto matcher = TopologicalMinorHeuristicSolver();

    // Check for minor relationship - expecting true because Q is a topological minor of G
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    auto correctMatching = std::vector<vertex>{1, 1, 1, 2, 3, 0};
    EXPECT_TRUE(utils::MatchingChecker::checkMinorMatching(G, Q, matching.value()));
    //  auto rempty = std::vector<vertex>();
    //    EXPECT_EQ(matching.value(), rempty);
}

TEST(InducedMinorIsomorphism, SmallNotInducedMinor) {
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

    // Define edges for the smaller graph Q (a cycle, which cannot be formed by contractions and deletions)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = InducedMinorHeuristic();

    // Check for induced minor relationship - expecting false because Q is a cycle but G is a chain
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(InducedMinorIsomorphism, SmallHasInducedMinor) {
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

    auto matcher = InducedMinorHeuristic();

    // Check for induced minor relationship - expecting true because Q can be derived from G
    EXPECT_TRUE(matcher.match(G, Q).has_value());
}

TEST(InducedTopologicalMinorIsomorphism, SmallNotInducedTopologicalMinor) {
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

    // Define edges for the smaller graph Q (a cycle, which cannot be induced as topological minor)
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = InducedTopologicalMinorHeuristicSolver();

    // Check for induced topological minor relationship - expecting false because Q is a cycle but G is a chain
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(InducedTopologicalMinorIsomorphism, SmallHasInducedButNotTopologicalMinor) {
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

    auto matcher = InducedTopologicalMinorHeuristicSolver();

    // Check for induced topological minor relationship - expecting true because Q can be induced as topological minor
    // from G
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

    auto matcher = TopologicalMinorHeuristicSolver(true);

    auto matching = matcher.match(G, topologicalMinor);
    EXPECT_TRUE(matching.has_value());
}

TEST(indcuedTopologicalMinor, random_130_vertex) {
    srand(SEED);

    auto topologicalMinor = utils::GraphFactory::random_connected_graph(100);
    auto G = utils::GraphFactory::random_edge_subdivisions(topologicalMinor, 30);

    auto matcher = InducedTopologicalMinorHeuristicSolver(true);

    auto matching = matcher.match(G, topologicalMinor);
    EXPECT_TRUE(matching.has_value());
}

TEST(inducedMinor, random_40_vertex) {
    srand(SEED);

    auto G = utils::GraphFactory::random_connected_graph(60, 0.9);
    auto inducedMinor = utils::GraphFactory::random_induced_minor(G, 8);

    auto matcher = InducedMinorHeuristic(true);

    auto matching = matcher.match(G, inducedMinor);
    EXPECT_TRUE(matching.has_value());
}

} // namespace pattern