#include "core.h"
#include <cstddef>
#include <vector>
#include <algorithm>

namespace core
{
Graph::Graph(int size) {
    _size = size;
    Graph::_adjacencyList = std::vector<std::vector<int>>(_size);
}

std::size_t Graph::size() const {
    return _size;
}

void Graph::add_edge(int u, int v) {
    // if (Graph::_adjacencyList[u] == nullptr) Graph::_adjacencyList[u] = std::vector<int>();
    Graph::_adjacencyList[u].push_back(v);
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
} // namespace core
