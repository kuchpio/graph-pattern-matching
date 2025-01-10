#include "induced_minor_heuristic.h"
#include <algorithm>

#define MAX_RECURSION_DEPTH 10

namespace pattern
{

std::optional<std::vector<vertex>> InducedMinorHeuristic::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;
    std::vector<vertex> mapping(G.size());
    std::iota(mapping.begin(), mapping.end(), 0);

    edges_ = G.edges();
    std::sort(edges_.begin(), edges_.end(),
              [](const std::tuple<vertex, vertex>& lhs, const std::tuple<vertex, vertex>& rhs) {
                  const auto lStart = std::get<0>(lhs);
                  const auto lEnd = std::get<1>(lhs);

                  const auto rStart = std::get<0>(rhs);
                  const auto rEnd = std::get<1>(rhs);

                  if (lStart != rStart) return lStart < rStart;

                  return lEnd < rEnd;
              });
    std::stable_sort(
        edges_.begin(), edges_.end(),
        [&G](const std::tuple<vertex, vertex>& lhs, const std::tuple<vertex, vertex>& rhs) {
            const auto lStart = std::get<0>(lhs);
            const auto lEnd = std::get<1>(lhs);

            const auto rStart = std::get<0>(rhs);
            const auto rEnd = std::get<1>(rhs);

            auto minLhs = std::min(G.degree_in(lStart) + G.degree_out(lStart), G.degree_in(lEnd) + G.degree_out(lEnd));
            auto minRhs = std::min(G.degree_in(rStart) + G.degree_out(rStart), G.degree_in(rEnd) + G.degree_out(rEnd));

            return minLhs < minRhs;
        });

    auto processedEdges = std::set<std::tuple<vertex, vertex>>();
    auto matching = inducedMinorRecursion(G, H, mapping, processedEdges, 0, 0);
    if (matching) return getResult(mapping_, matching.value());
    return std::nullopt;
}

std::optional<std::vector<vertex>> InducedMinorHeuristic::inducedMinorRecursion(
    const core::Graph& G, const core::Graph& H, const std::vector<vertex>& mapping,
    std::set<std::tuple<vertex, vertex>> processedEdges, int depth, int lastSkippedEdge) {
    if (depth > MAX_RECURSION_DEPTH) return std::nullopt;
    if (H.size() > G.size()) return std::nullopt;
    if (interrupted_) return std::nullopt;
    if (!maxDegreeConstraint(G, H)) return std::nullopt;

    auto subgraphMatching = subgraphMatcher_.get()->match(G, H);
    if (subgraphMatching) {
        mapping_ = mapping;
        return subgraphMatching;
    }

    for (int i = lastSkippedEdge; i < edges_.size(); i++) {
        auto [u, v] = edges_[i];
        if (!directed_) {
            if (processedEdges.contains(std::tie(v, u))) continue;
            processedEdges.insert(edges_[i]);
        }
        auto newMinor = contractEdge(G, mapping[u], mapping[v]);
        auto newMapping = updateMapping(mapping, mapping[u], mapping[v]);
        auto matching = inducedMinorRecursion(newMinor, H, newMapping, processedEdges, depth + 1, i + 1);
        if (matching) return matching;
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