#include "utils.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>
#include <unordered_map>

#define SEED 2000

namespace utils
{
core::Graph utils::GraphFactory::isomoporhic_graph(const core::Graph& G) {
    std::srand(unsigned(SEED));

    // Shuffle vertices
    std::vector<int> shuffled_vertices(G.size());
    std::iota(shuffled_vertices.begin(), shuffled_vertices.end(), 0);
    std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), std::default_random_engine{});

    // Create mapping
    std::unordered_map<int, int> mapping = std::unordered_map<int, int>();
    for (int i = 0; i < G.size(); i++) {
        mapping[i] = shuffled_vertices[i];
    }

    // Create new Graph
    core::Graph Q = core::Graph(G.size());
    for (int i = 0; i < G.size(); i++) {

        auto neighbours = G.get_neighbours(i);

        for (const auto& u : neighbours) {
            Q.add_edge(mapping[i], mapping[u]);
        }
    }
    return Q;
}

core::Graph utils::GraphFactory::random_graph(int n, float edge_propability) {
    core::Graph G = core::Graph(n);

    for (int i = 0; i < G.size(); i++) {
        for (int j = 0; j < G.size(); j++) {
            if (j == i) continue;
            float randomValue = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

            if (randomValue < edge_propability) {
                G.add_edge(i, j);
            }
        }
    }
    return G;
}
} // namespace utils