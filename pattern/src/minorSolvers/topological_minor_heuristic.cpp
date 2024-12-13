#include "topological_minor_heuristic.h"
namespace pattern
{
std::optional<std::vector<vertex>> TopologicalMinorHeuristic::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;
    auto subgraphMatching = subgraphSolver.match(G, H);
    if (subgraphMatching) return subgraphMatching;
}

} // namespace pattern