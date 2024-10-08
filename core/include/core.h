#pragma once
#include <vector>

namespace core
{
class Graph {
  private:
    int _size;
    std::vector<std::vector<int>> _adjacencyList;

  public:
    Graph(int size);
    void add_edge(int u, int v);
    bool remove_edge(int u, int v);
    int size() const;
    std::vector<int> get_neighbours(int v) const;
    int neighbours_count(int v) const;
};
} // namespace core
