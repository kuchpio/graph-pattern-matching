#include "core.h"
#include "native_isomorphism_matcher.h"
#include "utils.h"
#include "vf2_isomorphism_solver.hpp"
#include "gtest/gtest.h"
#include <optional>

namespace pattern
{

TEST(NativeGraphIsomorphism, random_100_vertex_graph) {

    core::Graph G = utils::GraphFactory::random_connected_graph(10, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    auto matcher = NativeIsomorphismMatcher();

    auto match = matcher.match(G, Q);

    EXPECT_EQ(match.has_value(), true);
}

TEST(Vf2GraphIsomorphism, small_graph_matching) {
    const std::size_t graphSize = 5;
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(graphSize);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(1, 3);
    G.add_edge(1, 4);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 0);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(2, 4);
    Q.add_edge(2, 0);
    Q.add_edge(3, 4);
    Q.add_edge(4, 0);

    auto matcher = Vf2IsomorphismSolver();

    auto match = matcher.match(G, Q);
    EXPECT_EQ(match.has_value(), true);
    bool checkMatching = utils::MatchingChecker::checkIsomorphismMatching(G, Q, match.value());

    EXPECT_EQ(checkMatching, true);
}
TEST(NativeGraphIsomorphism, small_graph_not_isomorphic) {
    std::size_t graphSize = 5;
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(graphSize);

    G.add_edge(0, 1);
    G.add_edge(0, 2);
    G.add_edge(1, 3);
    G.add_edge(2, 4);

    Q.add_edge(0, 1);
    Q.add_edge(0, 3);
    Q.add_edge(3, 2);
    Q.add_edge(2, 4);

    auto matcher = NativeIsomorphismMatcher();
    EXPECT_EQ(matcher.match(G, Q), std::nullopt);
}

TEST(Vf2GraphIsomorphism, small_graph_not_isomorphic) {
    std::size_t graphSize = 5;
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(graphSize);

    G.add_edge(0, 1);
    G.add_edge(0, 2);
    G.add_edge(1, 3);
    G.add_edge(2, 4);

    Q.add_edge(0, 1);
    Q.add_edge(0, 3);
    Q.add_edge(3, 2);
    Q.add_edge(2, 4);

    auto matcher = Vf2IsomorphismSolver();
    EXPECT_EQ(matcher.match(G, Q), std::nullopt);
}

TEST(NativeGraphIsomorphism, small_graph_matching) {
    const std::size_t graphSize = 5;
    core::Graph G = core::Graph(graphSize);
    core::Graph Q = core::Graph(graphSize);

    G.add_edge(0, 1);
    G.add_edge(1, 2);
    G.add_edge(1, 3);
    G.add_edge(1, 4);
    G.add_edge(2, 3);
    G.add_edge(3, 4);
    G.add_edge(4, 0);

    Q.add_edge(0, 1);
    Q.add_edge(1, 2);
    Q.add_edge(2, 3);
    Q.add_edge(2, 4);
    Q.add_edge(2, 0);
    Q.add_edge(3, 4);
    Q.add_edge(4, 0);

    auto matcher = NativeIsomorphismMatcher();

    auto match = matcher.match(G, Q);
    EXPECT_EQ(match.has_value(), true);
    bool checkMatching = utils::MatchingChecker::checkIsomorphismMatching(G, Q, match.value());

    EXPECT_EQ(checkMatching, true);
}

TEST(VF2GraphIsomorphism, random_100_vertex_graph) {

    core::Graph G = utils::GraphFactory::random_connected_graph(100, 0.4f);
    core::Graph Q = utils::GraphFactory::isomoporhic_graph(G);
    auto matcher = Vf2IsomorphismSolver();

    auto match = matcher.match(G, Q);

    EXPECT_EQ(match.has_value(), true);
}
} // namespace pattern