#include "core.h"
#include "pattern.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <optional>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

namespace pattern
{
bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int> Q_G_mapping,
                              std::unordered_map<int, int> G_Q_mapping, int v);
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}

bool is_sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return is_isomorphism(bigGraph, smallGraph);
}

bool is_isomorphism(const core::Graph& G, const core::Graph& Q) {

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

    for (auto v : vertex_indices) {
        if (G_Q_mapping.contains(v)) continue;
        if (is_isomorphism_recursion(G, Q, Q_G_mapping, G_Q_mapping, v)) return true;
    }
    return false;
}

bool is_isomorphism_recursion(const core::Graph& G, const core::Graph& Q, std::unordered_map<int, int> Q_G_mapping,
                              std::unordered_map<int, int> G_Q_mapping, int v) {

    if (Q_G_mapping.size() == G.size()) return true;

    // mamy juz cos przypisane

    // spróbuj dopisać v
    for (std::size_t u = 0; u < Q.size(); u++) {
        if (Q_G_mapping.contains(u)) continue;
        if (G.neighbours_count(v) != Q.neighbours_count(u)) continue;

        // dodaj u v
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

    // sprawdź czy nie koniec
    //  sprawdz ktore wierzcholki z mapowania nie maja wszystkich sasiadow, sproboj zmapowac pierwszego
    return false;
}

std::optional<std::size_t> match(const core::Graph& G, const core::Graph& Q,
                                 std::unordered_map<int, int>& Q_G_Q_G_mapping, int v) {

    // find unmapped vertex with his neighbours count

    return {}; // invalid match
}
} // namespace pattern
