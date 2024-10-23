#include "utils.h"
#include "core.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>
#include <unordered_map>

#define SEED 2000

namespace utils
{
void remove_empty_vertices(core::Graph& G);
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

std::vector<core::Graph> utils::GraphFactory::components(const core::Graph& G) {
    std::vector<bool> visited = std::vector<bool>(G.size());
    std::vector<core::Graph> components = std::vector<core::Graph>();

    std::stack<int> stack = std::stack<int>();
    for (int v = 0; v < G.size(); v++) {
        if (visited[v]) continue;

        std::vector<std::tuple<int, int>> edges = std::vector<std::tuple<int, int>>();
        stack.push(v);
        while (stack.empty() == false) {
            v = stack.top();
            stack.pop();
            visited[v] = true;
            for (auto neighbour : G.get_neighbours(v)) {
                if (visited[neighbour] == true) continue;
                stack.push(neighbour);
                edges.push_back(std::make_tuple(v, neighbour));
            }
        }
        auto Q = core::Graph(edges);
        remove_empty_vertices(Q);
        components.push_back(Q);
        edges.clear();
    }

    // sort components vector by size of graphs inside
    std::sort(components.begin(), components.end(),
              [](const core::Graph& G, const core::Graph& Q) { return G.size() > Q.size(); });

    return components;
}

std::vector<int> empty_vertices_indices(const std::vector<bool>& vec) {
    std::vector<int> false_vertices;

    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == false) false_vertices.push_back(i);
    }
    return false_vertices;
}

std::vector<int> get_empty_vertices(const core::Graph& G) {

    std::vector<bool> visited = std::vector<bool>(G.size());

    std::stack<int> stack = std::stack<int>();
    int v = 0;
    while (G.neighbours_count(v) == 0)
        v++;

    stack.push(v);

    while (stack.empty() == false) {
        int v = stack.top();
        stack.pop();
        visited[v] = true;

        for (auto neighbour : G.get_neighbours(v)) {
            if (visited[neighbour] == true) continue;
            stack.push(neighbour);
        }
    }
    return empty_vertices_indices(visited);
}

void remove_empty_vertices(core::Graph& G) {
    auto empty_vertices = get_empty_vertices(G);
    G.remove_vertices(empty_vertices);
}

} // namespace utils