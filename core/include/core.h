﻿#pragma once
#include <cstddef>
#include <vector>

namespace core
{
class Graph {
  private:
    std::size_t _size;
    std::vector<std::vector<int>> _adjacencyList;

  public:
    Graph(int size);
    void add_edge(int u, int v);
    bool remove_edge(int u, int v);
    std::size_t size() const;
    std::vector<int> get_neighbours(int v) const;
    int neighbours_count(int v) const;
};
} // namespace core
