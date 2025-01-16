#include "core.h"
#include "induced_subgraph_matcher.h"
#include "native_subgraph_matcher.h"
#include "utils.h"
#include "subgraph_matchers.h"
#include "gtest/gtest.h"

namespace pattern
{
TEST(NativeSubgraphIsomorphism, small_not_subgraph) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = NativeSubgraphMatcher();
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(Vf2SubgraphIsomorphism, small_not_subgraph) {
    std::size_t graphSize = 6;
    std::size_t subgraphSize = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(subgraphSize);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = Vf2SubgraphMatcher();
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(Vf2SubgraphIsomorphism, small_subgraph) {
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
    Vf2SubgraphMatcher matcher = Vf2SubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    EXPECT_TRUE(utils::MatchingChecker::checkSubgraphMatching(G, Q, matching.value()));
}

TEST(NativeSubgraphIsomorphism, small_subgraph) {
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
    auto matcher = NativeSubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    EXPECT_TRUE(utils::MatchingChecker::checkSubgraphMatching(G, Q, matching.value()));
}

TEST(NativeInducedSubgraphIsomorphism, small_not_induced_subgraph) {
    std::size_t graph_size = 6;
    std::size_t subgraph_size = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graph_size);
    core::Graph Q = core::Graph(subgraph_size);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = InducedSubgraphMatcher();
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(Vf2InducedSubgraphIsomorphism, small_not_induced_subgraph) {
    std::size_t graphSize = 6;
    std::size_t subgraphSize = 4;

    // Create the larger graph G
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(subgraphSize);

    // Define edges for the larger graph G
    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 5);

    // Define edges for the smaller subgraph Q
    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(3, 0);

    auto matcher = Vf2InducedSubgraphMatcher();
    EXPECT_FALSE(matcher.match(G, Q).has_value());
}

TEST(Vf2InducedSubgraphIsomorphism, small_induced_subgraph) {
    // Create the larger graph G
    std::size_t graphSize = 5;
    std::size_t subgraphSize = 3;

    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(subgraphSize);

    G.add_edge(0, 1);
    G.add_edge(0, 3);
    G.add_edge(0, 4);

    G.add_edge(1, 2);
    G.add_edge(1, 3);

    G.add_edge(2, 1);
    G.add_edge(2, 3);

    G.add_edge(3, 0);
    G.add_edge(3, 1);
    G.add_edge(3, 2);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 1);

    auto matcher = Vf2InducedSubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    EXPECT_TRUE(utils::MatchingChecker::checkInducedSubgraphMatching(G, Q, matching.value()));
}

TEST(NativeInducedSubgraphIsomorphism, small_induced_subgraph) {
    // Create the larger graph G
    std::size_t graphSize = 5;
    std::size_t subgraphSize = 3;

    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(subgraphSize);
    G.add_edge(0, 1);
    G.add_edge(0, 3);
    G.add_edge(0, 4);

    G.add_edge(1, 2);
    G.add_edge(1, 3);

    G.add_edge(2, 1);
    G.add_edge(2, 3);

    G.add_edge(3, 0);
    G.add_edge(3, 1);
    G.add_edge(3, 2);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 1);

    auto matcher = InducedSubgraphMatcher();
    auto matching = matcher.match(G, Q);
    EXPECT_TRUE(matching.has_value());

    EXPECT_TRUE(utils::MatchingChecker::checkInducedSubgraphMatching(G, Q, matching.value()));
}
} // namespace pattern