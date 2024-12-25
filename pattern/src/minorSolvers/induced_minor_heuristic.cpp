#include "induced_minor_heuristic.h"

#define MAX_RECURSION_DEPTH 10

namespace pattern
{

std::optional<std::vector<vertex>> InducedMinorHeuristic::tpRecursion(const core::Graph G, const core::Graph& H,
                                                                      const std::vector<vertex>& mapping, int depth,
                                                                      int lastSkippedEdge) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;
    if (interrupted_) return std::nullopt;
    if (!maxDegreeConstraint(G, H)) return std::nullopt;

    auto subgraphMatching = subgraphSolver.match(G, H);
    if (subgraphMatching) return subgraphMatching;

    for (int i = lastSkippedEdge; i < G.edges().size(); i++) {
        auto [u, v] = G.edges()[i];
        auto newMinor = contractEdge(G, u, v);
        auto newMapping = updateMapping(mapping, u, v);
        auto matching = tpRecursion(newMinor, H, newMapping, depth + 1, i);
        if (matching) return getResult(newMapping, matching.value());
    }
    return std::nullopt;
}

bool InducedMinorHeuristic::maxDegreeConstraint(const core::Graph& G, const core::Graph& H) {
    auto graphDegreesOut = G.degrees_out();
    auto minorDegreesOut = H.degrees_out();

    const std::size_t vertexDiff = G.size() - H.size();

    std::sort(graphDegreesOut.begin(), graphDegreesOut.end());
    std::sort(minorDegreesOut.begin(), minorDegreesOut.end());

    std::size_t index = 0;

    while (index < H.size()) {
        if (graphDegreesOut[index] > (minorDegreesOut[index] + vertexDiff)) return false;
        index++;
    }
    return true;
}
} // namespace pattern