#include "native_isomorphism_matcher.h"
#include <numeric>
#include "core.h"
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>
namespace pattern
{
std::optional<std::vector<vertex>> NativeIsomorphismMatcher::match(const core::Graph& bigGraph,
                                                                   const core::Graph& smallGraph) {
    return connectedIsomorphism(bigGraph, smallGraph);
}

std::optional<std::vector<vertex>> NativeIsomorphismMatcher::connectedIsomorphism(const core::Graph& G,
                                                                                  const core::Graph& Q) {

    if (G.size() != Q.size()) return std::nullopt;

    std::unordered_map<vertex, vertex> Q_G_mapping = std::unordered_map<vertex, vertex>();
    std::unordered_map<vertex, vertex> G_Q_mapping = std::unordered_map<vertex, vertex>();

    // start from vertex with max neighbours
    auto vertex_indices = std::vector<vertex>(G.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
    std::ranges::sort(vertex_indices, [&G](size_t i, size_t j) { return G.degree_out(i) > G.degree_out(j); });

    return isomorphismRecursion(G, Q, Q_G_mapping, G_Q_mapping, vertex_indices[0], std::nullopt);
}

bool checkNeighboursMapping(const core::Graph& G, const core::Graph& Q, std::unordered_map<vertex, vertex>& Q_G_mapping,
                            std::unordered_map<vertex, vertex>& G_Q_mapping, vertex v, vertex u) {
    // check negibours
    for (auto neighbor : G.get_neighbours(v)) {
        if (G_Q_mapping.contains(neighbor)) {
            if (!Q.has_edge(u, G_Q_mapping[neighbor])) return false;
        }
    }
    return true;
}

std::optional<std::vector<vertex>> NativeIsomorphismMatcher::isomorphismRecursion(
    const core::Graph& G, const core::Graph& Q, std::unordered_map<vertex, vertex>& Q_G_mapping,
    std::unordered_map<vertex, vertex>& G_Q_mapping, vertex v, std::optional<vertex> parent) {

    if (Q_G_mapping.size() == G.size()) return getMatching(Q_G_mapping);

    // find matching for v in Q
    for (vertex u = 0; u < Q.size(); u++) {
        if (Q_G_mapping.contains(u)) continue;
        if (G.degree_out(v) != Q.degree_out(u)) continue;
        if (!checkNeighboursMapping(G, Q, Q_G_mapping, G_Q_mapping, v, u)) continue;
        if (parent.has_value())
            if (!Q.has_edge(G_Q_mapping[parent.value()], u)) continue;

        Q_G_mapping.insert({u, v});
        G_Q_mapping.insert({v, u});

        if (Q_G_mapping.size() == G.size()) return getMatching(Q_G_mapping);

        // try function for each neighbour
        for (auto neighbour : G.get_neighbours(v)) {
            if (G_Q_mapping.contains(neighbour) == false) {
                auto matching = isomorphismRecursion(G, Q, Q_G_mapping, G_Q_mapping, neighbour, v);
                if (matching) return matching;
            }
        }

        Q_G_mapping.erase(u);
        G_Q_mapping.erase(v);
    }
    return std::nullopt;
}
} // namespace pattern