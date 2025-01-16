#include "topological_minor_exact.h"
#include <numeric>

#define MAX_RECURSION_DEPTH 1000

namespace pattern
{

std::optional<std::vector<vertex>> TopologicalMinorExact::minorRecursion(const core::Graph& G, const core::Graph& H,
                                                                         const std::vector<vertex>& mapping, int depth,
                                                                         int lastSkippedEdge) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;
    if (interrupted_) return std::nullopt;

    auto subgraphMatching = subgraphMatcher_->match(G, H);
    if (subgraphMatching) {
        mapping_ = mapping;
        return subgraphMatching;
    }

    for (int i = lastSkippedEdge; i < G.edges().size(); i++) {
        auto [u, v] = G.edges()[i];
        if ((G.degree_in(v) == maxDeegre_ && G.degree_out(v) == maxDeegre_) ||
            (G.degree_in(u) == maxDeegre_ && G.degree_out(u) == maxDeegre_)) {
            auto newMinor = contractEdge(G, u, v);
            auto newMapping = updateMapping(mapping, u, v);
            auto matching = minorRecursion(newMinor, H, newMapping, depth + 1, i);
            if (matching) return matching;
        }
    }
    return std::nullopt;
}
} // namespace pattern