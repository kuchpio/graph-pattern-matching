#include "induced_topological_minor_heuristic.h"

#define MAX_RECURSION_DEPTH 1000

namespace pattern
{
std::optional<std::vector<vertex>> InducedTopologicalMinorHeuristic::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;
    auto subgraphMatching = subgraphSolver.match(G, H);
    if (subgraphMatching) return subgraphMatching;

    return tpRecursion(G, H, 0);
}

std::optional<std::vector<vertex>> InducedTopologicalMinorHeuristic::tpRecursion(const core::Graph& G,
                                                                                 const core::Graph H, int depth) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;

    // iterate throug all edges
    for (auto [u, v] : H.edges()) {
        auto newMinor = subdivideEdge(H, u, v);
        auto subgraphMatching = subgraphSolver.match(G, H);
        if (subgraphMatching) return subgraphMatching;

        auto matching = tpRecursion(G, newMinor, depth + 1);
        if (matching) return matching;
    }
    return std::nullopt;
}

core::Graph InducedTopologicalMinorHeuristic::subdivideEdge(const core::Graph& G, vertex u, vertex v) {
    core::Graph Q = core::Graph(G);
    Q.subdivide_edge(u, v);
    return Q;
}

} // namespace pattern