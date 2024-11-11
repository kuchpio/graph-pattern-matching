#pragma once
#include "core.h"
#include <cstddef>
#include <vector>

namespace utils
{
class GraphFactory {
  private:
  public:
    static core::Graph isomoporhic_graph(const core::Graph& G);
    static core::Graph random_graph(std::size_t n, float edge_propability);
    static std::vector<core::Graph> components(const core::Graph& G);
    static core::Graph random_spanning_tree(std::size_t vertex_count);

  private:
    static std::vector<std::size_t> shuffled_vertices(std::size_t vertex_count, int seed);
    // static std::vector<core::Graph> components_with_reduced_vertices(const core::Graph& G);
};

} // namespace utils