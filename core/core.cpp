#include "core.h"
#include <cstddef>
#include <vector>
#include <algorithm>
#include <tuple>

namespace core
{
Graph::Graph(int size) {
    _size = size;
    Graph::_adjacencyList = std::vector<std::vector<int>>(_size);
}

Graph::Graph(const Graph& G) {
    _size = G._size;
    _adjacencyList = G._adjacencyList;
}

Graph::Graph(std::vector<std::tuple<int, int>> edges) {
    int max_value = -1;
    for (auto [u, v] : edges) {
        max_value = std::max({max_value, u, v});
    }

    _size = max_value + 1;
    _adjacencyList = std::vector<std::vector<int>>(_size);

    this->add_edges(edges);
}

std::size_t Graph::size() const {
    return _size;
}

void Graph::add_edge(int u, int v) {
    // if (Graph::_adjacencyList[u] == nullptr) Graph::_adjacencyList[u] = std::vector<int>();
    Graph::_adjacencyList[u].push_back(v);
}

bool Graph::has_edge(int u, int v) const {
    auto neighbours = this->get_neighbours(u);
    return std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end();
}

void Graph::add_edges(std::vector<std::tuple<int, int>> edges) {
    for (auto [u, v] : edges) {
        this->add_edge(u, v);
    }
}

std::vector<std::tuple<int, int>> Graph::edges() const {
    auto edges = std::vector<std::tuple<int, int>>();

    for (int i = 0; i < this->size(); i++) {
        for (auto neighbour : this->get_neighbours(i)) {
            edges.push_back(std::tie(i, neighbour));
        }
    }
    return edges;
}

bool Graph::remove_vertex(int v) {

    if (this->_size < v - 1) return false;

    // find vertex remove all edges, push back all edges from upcoming vertices
    for (int i = 0; i < this->size(); i++) {
        for (int neighbour : this->get_neighbours(i)) {
            if (neighbour > v) neighbour--;
            if (neighbour == v) this->remove_edge(i, neighbour);
        }
    }

    for (int i = v; i < this->size() - 1; i++) {
        this->_adjacencyList[i] = this->_adjacencyList[i + 1];
    }
    this->_size--;

    return true;
}

bool Graph::remove_vertices(std::vector<int> vertices) {
    // naive to be speed up
    for (auto v : vertices) {
        if (v - 1 > this->size()) return false;
    }

    for (auto v : vertices) {
        this->remove_vertex(v);
    }
    return true;
}

bool Graph::remove_edge(int u, int v) {
    std::vector<int> neighbours = Graph::_adjacencyList[u];
    auto v_iterator = std::find(neighbours.begin(), neighbours.end(), v);

    if (v_iterator != neighbours.end()) {
        neighbours.erase(v_iterator);
        return true;
    }
    return false;
}

int Graph::neighbours_count(int v) const {
    return Graph::_adjacencyList[v].size();
}

std::vector<int> Graph::get_neighbours(int v) const {
    return Graph::_adjacencyList[v];
}

std::size_t Graph::edge_count() const {
    std::size_t edge_count = 0;

    for (int i = 0; i < this->size(); i++) {
        edge_count += this->_adjacencyList[i].capacity();
    }
    return edge_count;
}

bool Graph::extract_edge(int u, int v) {
    // przesun wszystkich sasiadow v do u
    for (auto neighbour : this->get_neighbours(v)) {
        this->add_edge(u, neighbour);
    }
    return this->remove_vertex(v);
}

void Graph::topological_sort() {
}

} // namespace core
