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
    static core::Graph random_connected_graph(std::size_t vertex_count, float edge_probability = 0.4);
    static core::Graph random_minor(const core::Graph& G, std::size_t minorSize);
    static core::Graph random_edge_subdivisions(const core::Graph& G, std::size_t count);

  private:
    static std::vector<std::size_t> shuffled_vertices(std::size_t vertex_count);
    static void random_minor_operation(core::Graph& G, int v);
    // static std::vector<core::Graph> components_with_reduced_vertices(const core::Graph& G);
};

class MatchingChecker {
  public:
    static bool checkIsomorphismMatching(const core::Graph& G, const core::Graph& Q,
                                         const std::vector<vertex>& mapping);
    static bool checkSubgraphMatching(const core::Graph& G, const core::Graph& Q, const std::vector<vertex>& mapping);
    static bool checkInducedSubgraphMatching(const core::Graph& G, const core::Graph& Q,
                                             const std::vector<vertex>& mapping);
    static bool checkMinorMatching(const core::Graph& G, const core::Graph& H, const std::vector<vertex>& mapping);
    static bool checkInducedMinorMatching(const core::Graph& G, const core::Graph& H,
                                          const std::vector<vertex>& mapping);

  protected:
    static std::vector<std::vector<vertex>> toMinorMapping(const std::vector<vertex>& mapping);
};

} // namespace utils