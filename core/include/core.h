﻿#pragma once
#include <cstddef>
#include <vector>

typedef std::size_t vertex;

namespace core
{
class Graph {
  private:
    std::vector<std::vector<vertex>> _adjacencyList;

  public:
    Graph(const Graph& G);
    Graph(vertex size);
    Graph(std::vector<std::tuple<vertex, vertex>> edges);
    void add_edge(vertex u, vertex v);
    void add_edges(std::vector<std::tuple<vertex, vertex>> edges);
    bool remove_vertex(vertex v);
    bool remove_vertices(std::vector<vertex> vertices);
    bool remove_edge(vertex u, vertex v);
    bool has_edge(vertex u, vertex v) const;
    bool contract_edge(vertex u, vertex v);

    std::size_t degree_in(vertex v) const;
    vertex size() const;
    vertex edge_count() const;
    std::vector<std::tuple<vertex, vertex>> edges() const;
    std::vector<vertex> get_neighbours(vertex v) const;

    void topological_sort();
    std::size_t degree_out(vertex v) const;
};
} // namespace core

#include "../src/graph6Serializer.h"
