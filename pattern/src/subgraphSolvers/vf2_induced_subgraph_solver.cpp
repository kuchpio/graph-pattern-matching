#include "vf2_induced_subgraph_solver.hpp"

#include "argedit.h"
#include "argraph.h"
#include "core.h"
#include "vf2_sub_state.h"
#include <optional>
#include <vector>
#include "match.h"

Graph pattern::Vf2InducedSubgraphSolver::convert_graph(const core::Graph& G) {
    ARGEdit ed;

    for (vertex v = 0; v < G.size(); v++) {
        ed.InsertNode(&v);
    }

    for (auto [u, v] : G.edges()) {
        ed.InsertEdge(u, v, nullptr);
    }

    return Graph(&ed);
}

std::optional<std::vector<vertex>> pattern::Vf2InducedSubgraphSolver::match(const core::Graph& bigGraph,
                                                                            const core::Graph& smallGraph) {
    auto G = convert_graph(bigGraph);
    auto Q = convert_graph(smallGraph);

    int n;
    VF2SubState s0(&Q, &G);

    std::vector<node_id> big_nodes(smallGraph.size()), small_nodes(smallGraph.size());

    if (vf2::match(&s0, &n, big_nodes.data(), small_nodes.data())) {

        std::vector<vertex> result = std::vector<vertex>(smallGraph.size());
        for (int i = 0; i < result.size(); i++)
            result[i] = small_nodes[i];
        return result;
    }
    return std::nullopt;
}