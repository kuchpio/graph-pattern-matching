#pragma once
#include <cstddef>
#include <vector>

namespace core
{
class Graph {
  private:
    std::size_t _size;
    std::vector<std::vector<int>> _adjacencyList;

  public:
    Graph(const Graph& G);
    Graph(int size);
    Graph(std::vector<std::tuple<int, int>> edges);
    void add_edge(int u, int v);
    void add_edges(std::vector<std::tuple<int, int>> edges);
    bool remove_vertex(int v);
    bool remove_vertices(std::vector<int> vertices);
    bool remove_edge(int u, int v);
    bool has_edge(int u, int v) const;
    std::size_t size() const;
    std::size_t edge_count() const;
    std::vector<std::tuple<int, int>> edges() const;
    std::vector<int> get_neighbours(int v) const;
    int neighbours_count(int v) const;
};
} // namespace core
