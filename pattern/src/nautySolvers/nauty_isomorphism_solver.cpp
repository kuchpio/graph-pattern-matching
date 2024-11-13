#include "core.h"
#include "nauty_isomorphism_matcher.hpp"
#include <vector>

NTSparseGraph pattern::NautyIsomorphismMatcher::convert_graph(const core::Graph& G) {
    NTSparseGraph graph = NTSparseGraph(true, G.size());

    for (auto [u, v] : G.edges()) {
        graph.add_edge(u, v);
    }
    return graph;
}

bool pattern::NautyIsomorphismMatcher::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    auto G = convert_graph(bigGraph);
    auto Q = convert_graph(smallGraph);

    NautyTracesOptions nto;
    nto.get_canonical_node_order = true;

    NautyTracesResults ntr_G = traces(G, nto);
    NautyTracesResults ntr_Q = traces(Q, nto);
    std::vector<vertex> G_order = std::vector<vertex>(ntr_G.canonical_node_order.size());
    std::vector<vertex> Q_order = std::vector<vertex>(ntr_Q.canonical_node_order.size());

    for (auto i = 0; i < G_order.size(); i++) {
        G_order[i] = ntr_G.canonical_node_order[i];
        Q_order[i] = ntr_Q.canonical_node_order[i];
    }

    auto orderedG = bigGraph.reorder(G_order);
    auto orderedQ = smallGraph.reorder(Q_order);

    return orderedG == orderedQ;
}
