#include "core.h"
#include "nauty_isomorphism_matcher.hpp"

NTSparseGraph pattern::NautyIsomorphismMatcher::convert_graph(const core::Graph& G) {
    NTSparseGraph graph = NTSparseGraph(true);

    for (auto [u, v] : G.edges()) {
        graph.add_edge(u, v);
    }
    return graph;
}

bool pattern::NautyIsomorphismMatcher::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    auto G = convert_graph(bigGraph);
    auto Q = convert_graph(smallGraph);
}