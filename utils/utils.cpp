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
    std::vector<vertex> shuffled_vertices(G.size());
    std::iota(shuffled_vertices.begin(), shuffled_vertices.end(), 0);
    std::shuffle(shuffled_vertices.begin(), shuffled_vertices.end(), std::default_random_engine{});

    // Create mapping
    std::unordered_map<vertex, vertex> mapping = std::unordered_map<vertex, vertex>();
    for (vertex i = 0; i < G.size(); i++) {
        mapping[i] = shuffled_vertices[i];
    }

    // Create new Graph
    core::Graph Q = core::Graph(G.size());
    for (vertex i = 0; i < G.size(); i++) {

        auto neighbours = G.get_neighbours(i);

        for (const auto& u : neighbours) {
            Q.add_edge(mapping[i], mapping[u]);
        }
    }
    return Q;
}

core::Graph utils::GraphFactory::random_graph(std::size_t n, float edge_propability) {
    core::Graph G = core::Graph(n);

    for (vertex i = 0; i < G.size(); i++) {
        for (vertex j = 0; j < G.size(); j++) {
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

    std::stack<vertex> stack = std::stack<vertex>();
    for (vertex v = 0; v < G.size(); v++) {
        if (visited[v]) continue;

        std::vector<std::tuple<vertex, vertex>> edges = std::vector<std::tuple<vertex, vertex>>();
        stack.push(v);
        while (stack.empty() == false) {
            v = stack.top();
            stack.pop();
            visited[v] = true;
            for (auto neighbour : G.get_neighbours(v)) {
                if (visited[neighbour] == false) {
                    visited[neighbour] = true;
                    stack.push(neighbour);
                }
                edges.push_back(std::make_tuple(v, neighbour));
            }
        }
        if (edges.empty()) continue;
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

std::vector<vertex> empty_vertices_indices(const std::vector<bool>& vec) {
    std::vector<vertex> false_vertices;

    for (vertex i = 0; i < vec.size(); i++) {
        if (vec[i] == false) false_vertices.push_back(i);
    }
    return false_vertices;
}

std::vector<vertex> get_empty_vertices(const core::Graph& G) {

    std::vector<bool> visited = std::vector<bool>(G.size());

    std::stack<vertex> stack = std::stack<vertex>();
    vertex v = 0;
    while (G.degree_out(v) == 0)
        v++;

    stack.push(v);

    while (stack.empty() == false) {
        vertex v = stack.top();
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