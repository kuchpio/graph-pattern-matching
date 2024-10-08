#include "pattern.h"
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>
#include <unordered_map>

#define SEED 2000

int main() {
}

core::Graph create_isomoporhic_graph(const core::Graph& G) {
    std::srand(unsigned(SEED));

    // Shuffle vertices
    std::vector<int> shuffled_vertices(G.size());
    std::iota(shuffled_vertices.begin(), shuffled_vertices.end(), 0);
    std::random_shuffle(shuffled_vertices.begin(), shuffled_vertices.end());

    // Create mapping
    std::unordered_map<int, int> mapping;
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

bool small_graph_isomorphism_test() {
    return false;
}
