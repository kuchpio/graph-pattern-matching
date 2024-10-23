#include "core.h"
#include "pattern.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace pattern
{
bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int>& Q_G_mapping,
                              std::unordered_map<int, int>& G_Q_mapping, int v);

bool can_match_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                           const std::unordered_map<int, int>& mapping_big_small,
                           const std::unordered_map<int, int>& mapping_small_big, int v, int big_v);

bool match_isomorphism_components(std::vector<std::vector<core::Graph>>& G_components_by_size,
                                  std::vector<std::vector<core::Graph>>& Q_components_by_size);

bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}

bool sub_induced_isomorpshim(const core::Graph& bigGraph, const core::Graph& smallGraph) {

    if (bigGraph.size() == smallGraph.size()) return isomorphism(bigGraph, smallGraph);
    auto removed_vertices = std::vector<int>(bigGraph.size() - smallGraph.size());

    std::iota(removed_vertices.begin(), removed_vertices.end(), 0);

    do {
        core::Graph Q = core::Graph(bigGraph);
        Q.remove_vertices(removed_vertices);
        if (isomorphism(Q, smallGraph)) return true;

    } while (std::prev_permutation(removed_vertices.begin(), removed_vertices.end()));

    return false;
}

int find_first_unmapped(const core::Graph& G, std::unordered_map<int, int> map) {
    for (int i = 0; i < G.size(); i++) {
        if (!map.contains(i)) return i;
    }
    return -1;
}

bool sub_isomorphism_recursion(const core::Graph& bigGraph, const core::Graph& smallGraph,
                               std::unordered_map<int, int>& small_big_mapping,
                               std::unordered_map<int, int>& big_small_mapping, int v) {

    if (small_big_mapping.size() == smallGraph.size()) {
        return true;
    }

    for (std::size_t big_v = 0; big_v < bigGraph.size(); big_v++) {

        if (!can_match_isomorphism(bigGraph, smallGraph, big_small_mapping, small_big_mapping, v, big_v)) continue;
        big_small_mapping.insert({big_v, v});
        small_big_mapping.insert({v, big_v});

        if (small_big_mapping.size() == smallGraph.size()) {
            return true;
        }

        bool remaining_neighbours = false;
        for (auto neighbour : smallGraph.get_neighbours(v)) {
            if (small_big_mapping.contains(neighbour) == false) {
                remaining_neighbours = true;
                if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, neighbour))
                    return true;
            }
        }

        if (remaining_neighbours == false) {
            int first_unmapped = find_first_unmapped(smallGraph, small_big_mapping);
            if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, first_unmapped))
                return true;
        }
        small_big_mapping.erase(v);
        big_small_mapping.erase(big_v);
    }
    return false;
}

bool sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph) {

    std::unordered_map<int, int> small_big_mapping = std::unordered_map<int, int>();
    std::unordered_map<int, int> big_small_mapping = std::unordered_map<int, int>();

    auto vertex_indices = std::vector<int>(smallGraph.size());
    std::iota(vertex_indices.begin(), vertex_indices.end(), 0);

    for (auto vertex : vertex_indices) {
        if (sub_isomorphism_recursion(bigGraph, smallGraph, small_big_mapping, big_small_mapping, vertex_indices[0]))
            return true;
    }
    return false;
}

bool can_match_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                           const std::unordered_map<int, int>& mapping_big_small,
                           const std::unordered_map<int, int>& mapping_small_big, int v, int big_v) {
    if (mapping_big_small.contains(big_v)) return false;
    if (mapping_small_big.contains(v)) return false;

    for (auto neighbour : smallGraph.get_neighbours(v)) {
        if (mapping_small_big.contains(neighbour)) {
            if (bigGraph.has_edge(big_v, mapping_small_big.at(neighbour)) == false) return false;
        }
    }
    for (auto(pair) : mapping_small_big) {
        if (smallGraph.has_edge(pair.first, v) && bigGraph.has_edge(pair.second, big_v) == false) return false;
    }
    return true;
}

bool can_match_induced_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph,
                                   const std::unordered_map<int, int>& mapping_small_big,
                                   const std::unordered_map<int, int>& mapping_big_small, int v, int big_v) {
    if (mapping_small_big.contains(v)) return true;

    for (auto neighbour : smallGraph.get_neighbours(v)) {
        if (mapping_small_big.contains(neighbour)) {
            if (bigGraph.has_edge(big_v, mapping_small_big.at(neighbour)) == false) return false;
        }
    }

    for (auto neigbhour : bigGraph.get_neighbours(big_v)) {
        if (mapping_big_small.contains(neigbhour)) {
            if (smallGraph.has_edge(mapping_big_small.at(neigbhour), v) == false) return false;
        }
    }

    for (auto(pair) : mapping_small_big) {
        if (smallGraph.has_edge(pair.first, v) && bigGraph.has_edge(pair.second, big_v) == false) return false;
        if (bigGraph.has_edge(pair.second, big_v) && smallGraph.has_edge(pair.first, v)) return false;
    }
    return true;
}
bool isomorphism(const core::Graph& G, const core::Graph& Q) {
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
        if (G_components[i].size() != previous_size) {
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

bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int>& Q_G_mapping,
                              std::unordered_map<int, int>& G_Q_mapping, int v) {

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
