#include "ARGraph.hpp"
#include "FastCheck.hpp"
#include "core.h"
#include "vf3_subgraph_solver.hpp"

bool pattern::Vf3SubgraphSolver::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    auto G = convert_graph(bigGraph);
    auto Q = convert_graph(smallGraph);

    auto fastCheck = vflib::FastCheck<vertex, vertex, int, int>(&G, &Q);

    return fastCheck.CheckSubgraphIsomorphism();
}

vflib::ARGraph<vertex, int> pattern::Vf3SubgraphSolver::convert_graph(const core::Graph& G) {

    auto argLoader = vflib::ARGEdit<vertex, int>();
    auto direction = 1;
    for (vertex v = 0; v < G.size(); v++) {
        argLoader.InsertNode(v);
    }

    for (auto [u, v] : G.edges()) {
        argLoader.InsertEdge(u, v, direction);
    }

    auto graph = vflib::ARGraph<vertex, int>(&argLoader);

    return graph;
}