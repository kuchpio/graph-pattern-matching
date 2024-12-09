﻿#include "core.h"
#include <cstddef>
#include <vector>
#include <algorithm>
#include <tuple>

namespace core
{
Graph::Graph(vertex size) {
    Graph::_adjacencyList = std::vector<std::vector<vertex>>(size);
}

Graph::Graph(const Graph& G) {
    _adjacencyList = G._adjacencyList;
}

Graph::Graph(std::vector<std::tuple<vertex, vertex>> edges) {
    vertex max_value = 0;
    for (auto [u, v] : edges) {
        max_value = std::max({max_value, u, v});
    }

    vertex size = max_value + 1;
    _adjacencyList = std::vector<std::vector<vertex>>(size);

    this->add_edges(edges);
}

vertex Graph::size() const {
    return this->_adjacencyList.size();
}

vertex Graph::add_vertex() {
    this->_adjacencyList.emplace_back(std::vector<vertex>());
    return this->_adjacencyList.size() - 1;
}

void Graph::add_edge(vertex u, vertex v) {
    if (std::find(this->_adjacencyList[u].begin(), this->_adjacencyList[u].end(), v) != this->_adjacencyList[u].end())
        return;
    Graph::_adjacencyList[u].push_back(v);
}

bool Graph::has_edge(vertex u, vertex v) const {
    auto neighbours = this->get_neighbours(u);
    return std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end();
}

void Graph::add_edges(std::vector<std::tuple<vertex, vertex>> edges) {
    for (auto [u, v] : edges) {
        this->add_edge(u, v);
    }
}

std::vector<std::tuple<vertex, vertex>> Graph::edges() const {
    auto edges = std::vector<std::tuple<vertex, vertex>>();

    for (vertex i = 0; i < this->size(); i++) {
        for (auto neighbour : this->get_neighbours(i)) {
            edges.push_back(std::tie(i, neighbour));
        }
    }
    return edges;
}

bool Graph::remove_vertex(vertex v) {

    if (v > 0 && this->size() < (v - 1)) return false;

    // find vertex remove all edges, push back all edges from upcoming vertices
    for (vertex i = 0; i < this->size(); i++) {
        if (i == v) continue;
        if (std::find(this->_adjacencyList[i].begin(), this->_adjacencyList[i].end(), v) !=
            this->_adjacencyList[i].end())
            this->remove_edge(i, v);
        auto neighbours = this->_adjacencyList[i];
        for (vertex neighbour_index = 0; neighbour_index < neighbours.size(); neighbour_index++) {
            auto neighbour = this->_adjacencyList[i][neighbour_index];
            if (neighbour > v) this->_adjacencyList[i][neighbour_index]--;
        }
    }

    for (vertex i = v; i < this->size() - 1; i++) {
        this->_adjacencyList[i] = this->_adjacencyList[i + 1];
    }
    this->_adjacencyList.pop_back();

    return true;
}

bool Graph::remove_vertices(std::vector<vertex> vertices) {
    // naive to be speed up
    for (auto v : vertices) {
        if (v - 1 > this->size()) return false;
    }

    for (auto v : vertices) {
        this->remove_vertex(v);
    }
    return true;
}

bool Graph::remove_edge(vertex u, vertex v) {
    auto v_iterator = std::find(this->_adjacencyList[u].begin(), this->_adjacencyList[u].end(), v);
    if (v_iterator == this->_adjacencyList[u].end()) return false;

    this->_adjacencyList[u].erase(v_iterator);
    return true;
}

vertex Graph::degree_out(vertex v) const {
    return Graph::_adjacencyList[v].size();
}

std::vector<vertex> Graph::get_neighbours(vertex v) const {
    if (v >= this->size()) return std::vector<vertex>();
    return Graph::_adjacencyList[v];
}

vertex Graph::edge_count() const {
    vertex edge_count = 0;

    for (vertex i = 0; i < this->size(); i++) {
        edge_count += this->_adjacencyList[i].capacity();
    }
    return edge_count;
}

bool Graph::contract_edge(vertex u, vertex v) {
    // przesun wszystkich sasiadow v do u
    for (auto neighbour : this->get_neighbours(v)) {
        if (neighbour == u) continue;
        this->add_edge(u, neighbour);
    }

    // i wszystkich wchodzacydh do v do u
    for (vertex i = 0; i < this->size(); i++) {
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

vertex Graph::degree_in(vertex v) const {

    std::size_t degree_in = 0;
    for (vertex i = 0; i < this->size(); i++) {
        if (i == v) continue;
        auto neighbours = this->_adjacencyList[i];
        if (std::find(neighbours.begin(), neighbours.end(), v) != neighbours.end()) degree_in++;
    }
    return degree_in;
}

Graph Graph::applyMapping(const std::vector<vertex>& mapping) const {
    Graph G = Graph(this->size());
    for (auto [u, v] : this->edges()) {
        G.add_edge(mapping[u], mapping[v]);
    }
    return G;
}

void Graph::reorder(const std::vector<vertex>& order) {

    std::vector<vertex> mapping = std::vector<vertex>(order.size());
    for (int i = 0; i < mapping.size(); i++) {
        mapping[order[i]] = i;
    }

    auto G = this->applyMapping(mapping);

    this->_adjacencyList = G._adjacencyList;
}

Graph Graph::reorder(const std::vector<vertex>& order) const {
    std::vector<vertex> mapping = std::vector<vertex>(order.size());
    for (int i = 0; i < mapping.size(); i++) {
        mapping[order[i]] = i;
    }

    auto G = this->applyMapping(mapping);

    return G;
}

bool Graph::operator==(const core::Graph& G) const {
    return this->_adjacencyList == G._adjacencyList;
}

bool Graph::is_subgraph(const core::Graph& subgprah) const {
    for (auto v = 0; v < subgprah.size(); v++) {
        for (auto u : subgprah.get_neighbours(v)) {
            if (this->has_edge(v, u) == false) return false;
        }
    }
    return true;
}

bool Graph::is_induced_subgraph(const core::Graph& subgprah) const {
    for (vertex v = 0; v < subgprah.size(); v++) {
        for (vertex u = 0; u < subgprah.size(); u++) {
            if (subgprah.has_edge(u, v))
                if (this->has_edge(u, v) == false) return false;
            if (this->has_edge(u, v))
                if (subgprah.has_edge(u, v) == false) return false;
        }
    }
    return true;
}

} // namespace core
