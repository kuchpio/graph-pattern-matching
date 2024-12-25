#include "induced_minor_heuristic.h"

#define MAX_RECURSION_DEPTH 10

namespace pattern
{

std::optional<std::vector<vertex>> InducedMinorHeuristic::tpRecursion(const core::Graph G, const core::Graph& H,
                                                                      const std::vector<vertex>& mapping, int depth,
                                                                      int lastSkippedEdge) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;

    auto subgraphMatching = subgraphSolver.match(G, H);
    if (subgraphMatching) return subgraphMatching;
    if (interrupted_) return std::nullopt;

    for (int i = lastSkippedEdge; i < G.edges().size(); i++) {
        auto [u, v] = G.edges()[i];
        auto newMinor = contractEdge(G, u, v);
        auto newMapping = updateMapping(mapping, u, v);
        auto matching = tpRecursion(newMinor, H, newMapping, depth + 1, i);
        if (matching) return getResult(newMapping, matching.value());
    }
    return std::nullopt;
}
} // namespace pattern