#include "topological_minor_heuristic.h"
#include <numeric>

#define MAX_RECURSION_DEPTH 1000

namespace pattern
{
std::optional<std::vector<vertex>> TopologicalMinorHeuristic::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;
    std::vector<vertex> mapping(G.size());
    std::iota(mapping.begin(), mapping.end(), 0);

    auto matching = tpRecursion(G, H, mapping, 0, 0);
    if (matching) return getResult(mapping_, matching.value());
    return std::nullopt;
}

std::optional<std::vector<vertex>> TopologicalMinorHeuristic::tpRecursion(const core::Graph G, const core::Graph& H,
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
<<<<<<< HEAD
        if (G.degree_in(v) + G.degree_out(v) == 2 || G.degree_in(u) + G.degree_out(u) == 2) {
=======
        if ((G.degree_in(v) == 2 && G.degree_out(v) == 2) || (G.degree_in(u) == 2 && G.degree_out(u) == 2)) {
>>>>>>> master
            auto newMinor = contractEdge(G, u, v);
            auto newMapping = updateMapping(mapping, u, v);
            auto matching = tpRecursion(newMinor, H, newMapping, depth + 1, i);
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

std::vector<vertex> TopologicalMinorHeuristic::updateMapping(const std::vector<vertex>& mapping, vertex u, vertex v) {
    auto newMapping = std::vector<vertex>(mapping);
    for (vertex& vertex : newMapping) {
        if (vertex == v)
            vertex = u;
        else if (vertex > v)
            vertex--;
    }
    return newMapping;
}

std::vector<vertex> TopologicalMinorHeuristic::getResult(const std::vector<vertex>& mapping,
                                                         const std::vector<vertex>& contractedResult) {
    auto result = std::vector<vertex>(mapping);
    for (vertex& v : result) {
        v = contractedResult[v];
    }
    return result;
}
} // namespace pattern