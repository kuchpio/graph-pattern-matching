#include "utils.h"
#include "core.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

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

    return G.reorder(shuffled_vertices);
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

core::Graph GraphFactory::random_spanning_tree(std::size_t vertex_count) {
    core::Graph SpanningTree = core::Graph(vertex_count);

    auto vertices = shuffled_vertices(vertex_count);
    std::vector<bool> visited = std::vector<bool>(vertex_count);

    auto current_vertex = vertices.back();
    vertices.pop_back();
    visited[current_vertex] = true;

    while (!vertices.empty()) {
        auto neighbour = vertices.back();
        vertices.pop_back();
        if (!visited[neighbour]) {
            SpanningTree.add_edge(current_vertex, neighbour);
            visited[neighbour] = true;
        }
    }
    return SpanningTree;
}

core::Graph GraphFactory::random_connected_graph(std::size_t vertex_count, float edge_probability) {
    auto spanningTree = random_spanning_tree(vertex_count);

    srand(SEED);

    for (int v = 0; v < spanningTree.size(); v++) {
        for (int u = 0; u < spanningTree.size(); u++) {
            if (u == v) continue;
            float probability = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            if (probability > edge_probability) {
                spanningTree.add_edge(v, u);
            }
        }
    }
    return spanningTree;
}

core::Graph GraphFactory::random_minor(core::Graph G, std::size_t minorSize) {

    while (G.size() > minorSize) {
        std::size_t randomVertex = rand() % G.size();
        random_minor_operation(G, randomVertex);
    }
    return G;
}

std::size_t random_neighbour(const core::Graph& G, int v) {
    auto neighbours = G.get_neighbours(v);
    std::size_t random_index = rand() % neighbours.size();
    return neighbours[random_index];
}

void GraphFactory::random_minor_operation(core::Graph& G, int v) {
    // choose random operation
    int random_operation = rand() % 3;

    std::size_t randomNeighbour;
    switch (random_operation) {
    case 0:
        G.remove_vertex(v);
        break;
    case 1:
        randomNeighbour = random_neighbour(G, v);
        G.remove_edge(v, randomNeighbour);
        break;
    case 2:
        randomNeighbour = random_neighbour(G, v);
        G.contract_edge(v, randomNeighbour);
        break;
    }
}

std::vector<std::size_t> GraphFactory::shuffled_vertices(std::size_t vertex_count) {
    std::vector<std::size_t> vertices = std::vector<std::size_t>(vertex_count);
    std::iota(vertices.begin(), vertices.end(), 0);

    std::shuffle(vertices.begin(), vertices.end(), std::mt19937());
    return vertices;
}

bool MatchingChecker::checkIsomorphismMatching(const core::Graph& G, const core::Graph& Q,
                                               const std::vector<vertex>& mapping) {
    auto reorderedQ = Q.applyMapping(mapping);

    return reorderedQ == G;
}

} // namespace utils