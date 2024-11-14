#include "vf2_sub_state.h"
#include "vf2_subgraph_solver.hpp"

#include "argedit.h"
#include "argraph.h"
#include "core.h"
#include <vector>

#include "match.h"

Graph pattern::Vf2SubgraphSolver::convert_graph(const core::Graph& G) {
    ARGEdit ed;

    for (vertex v = 0; v < G.size(); v++) {
        ed.InsertNode(&v);
    }

    for (auto [u, v] : G.edges()) {
        ed.InsertEdge(u, v, nullptr);
    }

    return Graph(&ed);
}

bool pattern::Vf2SubgraphSolver::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    auto G = convert_graph(bigGraph);
    auto Q = convert_graph(smallGraph);

    int n;
    VF2SubState s0(&Q, &G);

    std::vector<node_id> big_nodes(smallGraph.size()), small_nodes(smallGraph.size());

    return vf2::match(&s0, &n, big_nodes.data(), small_nodes.data());
}