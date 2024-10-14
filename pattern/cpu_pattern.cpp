#include "pattern.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace pattern
{
bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int> Q_G_mapping,
                              std::unordered_map<int, int> G_Q_mapping, int v);

bool match_isomorphism_components(std::vector<std::vector<core::Graph>>& G_components_by_size,
                                  std::vector<std::vector<core::Graph>>& Q_components_by_size);

bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}

bool is_sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return connected_isomorphism(bigGraph, smallGraph);
}

bool check_isomorphism(const core::Graph& G, const core::Graph& Q) {
    if (G.size() != Q.size()) return false;
    auto G_components = utils::GraphFactory::components(G);
    auto Q_components = utils::GraphFactory::components(Q);

    if (G_components.size() != Q_components.size()) return false;

    auto G_components_by_size = std::vector<std::vector<core::Graph>>();
    auto Q_components_by_size = std::vector<std::vector<core::Graph>>();

    int previous_size = G_components[0].size();
    auto current_G_components = std::vector<core::Graph>();
    auto current_Q_components = std::vector<core::Graph>();

    for (int i = 0; i < G_components.size(); i++) {

        if (G_components[i].size() != Q_components[i].size()) return false;
        if (G_components.size() != previous_size) {
            G_components_by_size.push_back(current_G_components);
            Q_components_by_size.push_back(current_Q_components);
            current_G_components.clear();
            current_Q_components.clear();
        }
        current_G_components.push_back(G_components[i]);
        current_Q_components.push_back(Q_components[i]);
    }

    // try to match every
    return match_isomorphism_components(G_components_by_size, Q_components_by_size);
}

bool match_isomorphism_components(std::vector<std::vector<core::Graph>>& G_components_by_size,
                                  std::vector<std::vector<core::Graph>>& Q_components_by_size) {
    for (int i = 0; i < G_components_by_size.size(); i++) {
        bool match = false;
        for (const auto& G : G_components_by_size[i]) {
            for (const auto& Q : Q_components_by_size[i]) {
                if (connected_isomorphism(G, Q)) match = true;
            }
        }
        if (match == false) return false;
    }
    return true;
}

bool connected_isomorphism(const core::Graph& G, const core::Graph& Q) {

    if (G.size() != Q.size()) return false;

    std::unordered_map<int, int> Q_G_mapping = std::unordered_map<int, int>();
    std::unordered_map<int, int> G_Q_mapping = std::unordered_map<int, int>();
    // bierzemy pierwszy wierzchołek
    // znajdz wierzcholek ktory maksymalizuje n(v)
    auto vertex_indices = std::vector<int>(G.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);

    // Find the row index with the maximum number of ones
    std::ranges::sort(vertex_indices, [&G](size_t i, size_t j) {
        return G.neighbours_count(i) > G.neighbours_count(j); // Sort by descending count of 1s
    });
    return is_isomorphism_recursion(G, Q, Q_G_mapping, G_Q_mapping, vertex_indices[0]);
}

bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int> Q_G_mapping,
                              std::unordered_map<int, int> G_Q_mapping, int v) {

    if (Q_G_mapping.size() == G.size()) return true;

    // find matching for v in Q
    for (std::size_t u = 0; u < Q.size(); u++) {
        if (Q_G_mapping.contains(u)) continue;
        if (G.neighbours_count(v) != Q.neighbours_count(u)) continue;

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
} // namespace pattern
