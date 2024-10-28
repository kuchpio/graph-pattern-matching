#include "core.h"
#include <cstddef>
#include <vector>
#include <algorithm>
#include <tuple>

namespace core
{
Graph::Graph(int size) {
    Graph::_adjacencyList = std::vector<std::vector<int>>(size);
}

Graph::Graph(const Graph& G) {
    _adjacencyList = G._adjacencyList;
}

Graph::Graph(std::vector<std::tuple<int, int>> edges) {
    int max_value = -1;
    for (auto [u, v] : edges) {
        max_value = std::max({max_value, u, v});
    }

    int size = max_value + 1;
    _adjacencyList = std::vector<std::vector<int>>(size);

    this->add_edges(edges);
}

std::size_t Graph::size() const {
    return this->_adjacencyList.size();
}

void Graph::add_edge(int u, int v) {
    if (std::find(this->_adjacencyList[u].begin(), this->_adjacencyList[u].end(), v) != this->_adjacencyList[u].end())
        return;
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

    if (v > 0 && this->size() < (v - 1)) return false;

    // find vertex remove all edges, push back all edges from upcoming vertices
    for (int i = 0; i < this->size(); i++) {
        if (i == v) continue;
        if (std::find(this->_adjacencyList[i].begin(), this->_adjacencyList[i].end(), v) !=
            this->_adjacencyList[i].end())
            this->remove_edge(i, v);
        auto neighbours = this->_adjacencyList[i];
        for (int neighbour_index = 0; neighbour_index < neighbours.size(); neighbour_index++) {
            auto neighbour = this->_adjacencyList[i][neighbour_index];
            if (neighbour > v) this->_adjacencyList[i][neighbour_index]--;
        }
    }

    for (int i = v; i < this->size() - 1; i++) {
        this->_adjacencyList[i] = this->_adjacencyList[i + 1];
    }
    this->_adjacencyList.pop_back();

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
    auto v_iterator = std::find(this->_adjacencyList[u].begin(), this->_adjacencyList[u].end(), v);
    if (v_iterator == this->_adjacencyList[u].end()) return false;

    this->_adjacencyList[u].erase(v_iterator);

    if (this->degree_in(u) == 0 && this->degree_out(u) == 0) this->remove_vertex(u);

    return true;
}

int Graph::degree_out(int v) const {
    return Graph::_adjacencyList[v].size();
}

std::vector<int> Graph::get_neighbours(int v) const {
    if (v >= this->size()) return std::vector<int>();
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
        if (neighbour == u) continue;
        this->add_edge(u, neighbour);
    }

    // i wszystkich wchodzacydh do v do u
    for (int i = 0; i < this->size(); i++) {
        if (i == v || i == u) continue;
        for (auto neighbour : this->get_neighbours(i)) {
            if (neighbour == v) {
                this->add_edge(i, u);
                break;
            }
        }
    }
    auto succcess = this->remove_vertex(v);

    return succcess;
}

void Graph::topological_sort() {
}

int Graph::degree_in(int v) const {

    int degree_in = 0;
    for (int i = 0; i < this->size(); i++) {
        if (i == v) continue;
        auto neighbours = this->_adjacencyList[i];
        if (std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end()) degree_in++;
    }
    return degree_in;
}

} // namespace core
