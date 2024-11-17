#include "isomorphism_matcher.h"
#include <numeric>
#include "core.h"
#include "utils.h"
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>
namespace pattern
{
bool IsomorphismMatcher::match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    if (bigGraph.size() != smallGraph.size()) return false;
    auto G_components = utils::GraphFactory::components(bigGraph);
    auto Q_components = utils::GraphFactory::components(smallGraph);

    if (G_components.size() != Q_components.size()) return false;

    auto G_components_by_size = std::vector<std::vector<core::Graph>>();
    auto Q_components_by_size = std::vector<std::vector<core::Graph>>();

    vertex previous_size = G_components[0].size();
    auto current_G_components = std::vector<core::Graph>();
    auto current_Q_components = std::vector<core::Graph>();

    for (vertex i = 0; i < G_components.size(); i++) {
        if (G_components[i].size() != Q_components[i].size()) return false;
        if (G_components[i].size() != previous_size) {
            G_components_by_size.push_back(current_G_components);
            Q_components_by_size.push_back(current_Q_components);
            current_G_components.clear();
            current_Q_components.clear();
        }
        current_G_components.push_back(G_components[i]);
        current_Q_components.push_back(Q_components[i]);
    }
    G_components_by_size.push_back(current_G_components);
    Q_components_by_size.push_back(current_Q_components);

    // try to match every
    return match_isomorphism_components(G_components_by_size, Q_components_by_size);
}

bool IsomorphismMatcher::match_isomorphism_components(std::vector<std::vector<core::Graph>>& G_components_by_size,
                                                      std::vector<std::vector<core::Graph>>& Q_components_by_size) {
    for (std::size_t size_index = 0; size_index < G_components_by_size.size(); size_index++) {
        for (const auto& G : G_components_by_size[size_index]) {
            bool match = false;

            for (auto Q_it = Q_components_by_size[size_index].begin(); Q_it != Q_components_by_size[size_index].end();
                 Q_it++) {
                if (connected_isomorphism(G, *Q_it)) {
                    match = true;
                    Q_components_by_size[size_index].erase(Q_it);
                    break;
                }
            }
            if (match == false) return false;
        }
    }
    return true;
}

bool IsomorphismMatcher::connected_isomorphism(const core::Graph& G, const core::Graph& Q) {

    if (G.size() != Q.size()) return false;

    std::unordered_map<vertex, vertex> Q_G_mapping = std::unordered_map<vertex, vertex>();
    std::unordered_map<vertex, vertex> G_Q_mapping = std::unordered_map<vertex, vertex>();
    // bierzemy pierwszy wierzchołek
    // znajdz wierzcholek ktory maksymalizuje n(v)
    auto vertex_indices = std::vector<vertex>(G.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);

    // Find the row index with the maximum number of ones
    std::ranges::sort(vertex_indices, [&G](size_t i, size_t j) {
        return G.degree_out(i) > G.degree_out(j); // Sort by descending count of 1s
    });
    return is_isomorphism_recursion(G, Q, Q_G_mapping, G_Q_mapping, vertex_indices[0]);
}

bool IsomorphismMatcher::is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q,
                                                  std::unordered_map<vertex, vertex>& Q_G_mapping,
                                                  std::unordered_map<vertex, vertex>& G_Q_mapping, vertex v) {

    if (Q_G_mapping.size() == G.size()) return true;

    // find matching for v in Q
    for (vertex u = 0; u < Q.size(); u++) {
        if (Q_G_mapping.contains(u)) continue;
        if (G.degree_out(v) != Q.degree_out(u)) continue;

        Q_G_mapping.insert({u, v});
        G_Q_mapping.insert({v, u});

        if (Q_G_mapping.size() == G.size()) return true;

        // try function for each neighbour
        for (auto neighbour : G.get_neighbours(v)) {
            if (G_Q_mapping.contains(neighbour) == false)
                if (is_isomorphism_recursion(G, Q, Q_G_mapping, G_Q_mapping, neighbour)) return true;
        }

        Q_G_mapping.erase(u);
        G_Q_mapping.erase(v);
    }
    return false;
}

std::optional<std::vector<vertex>> IsomorphismMatcher::connectedIsomorphism(const core::Graph& G,
                                                                            const core::Graph& Q) {

    if (G.size() != Q.size()) return std::nullopt;

    std::unordered_map<vertex, vertex> Q_G_mapping = std::unordered_map<vertex, vertex>();
    std::unordered_map<vertex, vertex> G_Q_mapping = std::unordered_map<vertex, vertex>();

    // start from vertex with max neighbours
    auto vertex_indices = std::vector<vertex>(G.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
    std::ranges::sort(vertex_indices, [&G](size_t i, size_t j) { return G.degree_out(i) > G.degree_out(j); });

    return isomorphismRecursion(G, Q, Q_G_mapping, G_Q_mapping, vertex_indices[0]);
}

std::optional<std::vector<vertex>> IsomorphismMatcher::isomorphismRecursion(
    const core::Graph& G, const core::Graph& Q, std::unordered_map<vertex, vertex>& Q_G_mapping,
    std::unordered_map<vertex, vertex>& G_Q_mapping, vertex v) {

    if (Q_G_mapping.size() == G.size()) return getMatching(Q_G_mapping);

    // find matching for v in Q
    for (vertex u = 0; u < Q.size(); u++) {
        if (Q_G_mapping.contains(u)) continue;
        if (G.degree_out(v) != Q.degree_out(u)) continue;

        Q_G_mapping.insert({u, v});
        G_Q_mapping.insert({v, u});

        if (Q_G_mapping.size() == G.size()) return getMatching(Q_G_mapping);

        // try function for each neighbour
        for (auto neighbour : G.get_neighbours(v)) {
            if (G_Q_mapping.contains(neighbour) == false)
                if (is_isomorphism_recursion(G, Q, Q_G_mapping, G_Q_mapping, neighbour))
                    return getMatching(Q_G_mapping);
        }

        Q_G_mapping.erase(u);
        G_Q_mapping.erase(v);
    }
    return std::nullopt;
}

std::vector<vertex> IsomorphismMatcher::getMatching(std::unordered_map<vertex, vertex> mapping) {
    std::vector<vertex> matching = std::vector<vertex>(mapping.size());
    for (const auto& pair : mapping) {
        matching[pair.first] = pair.second;
    }
    return matching;
}

} // namespace pattern