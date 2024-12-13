#pragma once
#include <cstddef>
#include <vector>
#include <tuple>

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
    vertex add_vertex();
    void add_edge(vertex u, vertex v);
    void add_edges(std::vector<std::tuple<vertex, vertex>> edges);
    bool remove_vertex(vertex v);
    bool remove_vertices(std::vector<vertex> vertices);
    bool remove_edge(vertex u, vertex v);
    bool has_edge(vertex u, vertex v) const;
    bool contract_edge(vertex u, vertex v);
    bool is_subgraph(const core::Graph& G) const;
    bool is_induced_subgraph(const core::Graph& G) const;
    vertex subdivide_edge(vertex u, vertex v);

    // operator
    bool operator==(const core::Graph& G) const;

    std::size_t degree_in(vertex v) const;
    vertex size() const;
    vertex edge_count() const;
    std::vector<std::tuple<vertex, vertex>> edges() const;
    std::vector<vertex> get_neighbours(vertex v) const;

    void topological_sort();
    void reorder(const std::vector<vertex>& order);
    Graph applyMapping(const std::vector<vertex>& mapping) const;
    Graph reorder(const std::vector<vertex>& order) const;
    std::size_t degree_out(vertex v) const;
};
} // namespace core

#include "../src/graph6Serializer.h"
