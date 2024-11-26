#pragma once
#include "core.h"
#include "native_minor_matcher.h"
namespace pattern
{
class TopologicalInducedMinorMatcher : public NativeMinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q);

  private:
    bool topological_induced_minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                             std::optional<vertex> last_neighbour_index);
    std::optional<std::vector<vertex>> topologicalInducedMinorRecursion(const core::Graph& G, const core::Graph& H,
                                                                        vertex v,
                                                                        std::optional<vertex> last_neighbour_index);
};
} // namespace pattern