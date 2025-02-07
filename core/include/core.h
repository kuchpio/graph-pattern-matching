﻿#pragma once
#include <cstddef>
#include <vector>
#include <tuple>
#include <optional>

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
    bool remove_vertices(const std::vector<vertex>& verticesSortedDesc);
    bool remove_vertices(const std::vector<vertex>& verticesSortedDesc, const int* toBeRemoved,
                         const vertex* vertexIndexDelta);
    bool remove_edge(vertex u, vertex v);
    bool has_edge(vertex u, vertex v) const;
    bool contract_edge(vertex u, vertex v);
    bool has_subgraph(const core::Graph& G) const;
    bool is_induced_subgraph(const core::Graph& G) const;
    bool connected() const;
    core::Graph undirected() const;
    vertex subdivide_edge(vertex u, vertex v);

    // operator
    bool operator==(const core::Graph& G) const;

    std::size_t degree_in(vertex v) const;
    vertex size() const;
    vertex edge_count() const;
    std::vector<std::tuple<vertex, vertex>> edges() const;
    std::vector<vertex> get_neighbours(vertex v) const;
    const std::vector<vertex>& neighbours(vertex v) const;

    void topological_sort();
    void reorder(const std::vector<vertex>& order);
    Graph applyMapping(const std::vector<vertex>& mapping) const;
    Graph reorder(const std::vector<vertex>& order) const;
    std::size_t degree_out(vertex v) const;
    std::vector<std::size_t> degrees_out() const;
};

class IPatternMatcher {
  public:
    virtual std::optional<std::vector<vertex>> match(const core::Graph& searchSpace, const core::Graph& pattern) = 0;
    virtual void interrupt() = 0;
    virtual ~IPatternMatcher() = default;
};

} // namespace core
