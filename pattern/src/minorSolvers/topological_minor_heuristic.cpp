#include "topological_minor_heuristic.h"

#define MAX_RECURSION_DEPTH 1000

namespace pattern
{
std::optional<std::vector<vertex>> TopologicalMinorHeuristic::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;
    return tpRecursion(G, H, 0);
}

std::optional<std::vector<vertex>> TopologicalMinorHeuristic::tpRecursion(const core::Graph G, const core::Graph& H,
                                                                          int depth) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;

    auto subgraphMatching = subgraphMatcher_->match(G, H);
    if (subgraphMatching) return subgraphMatching;

    // iterate throug all edges
    for (auto [u, v] : G.edges()) {
        if(G.degree_in(v) + G.degree_out(v) == 2) {
            auto newMinor = contractEdge(G, u, v);
            auto matching = tpRecursion(G, newMinor, depth + 1);
            if (matching) return matching;
        }
    }
    return std::nullopt;
}

core::Graph TopologicalMinorHeuristic::contractEdge(const core::Graph& G, vertex u, vertex v) {
    core::Graph Q = core::Graph(G);
    Q.contract_edge(u, v);
    return Q;
}
} // namespace pattern