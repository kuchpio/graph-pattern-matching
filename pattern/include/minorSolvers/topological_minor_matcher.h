#pragma once
#include "core.h"
#include "native_minor_matcher.h"
namespace pattern
{
class TopologicalMinorMatcher : public NativeMinorMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H);
    std::optional<std::vector<vertex>> matching(const core::Graph& G, const core::Graph& Q);

  private:
    bool topological_minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                     std::optional<vertex> last_neighbour_index);
    std::optional<std::vector<vertex>> topologicalMinorRecursion(const core::Graph& G, const core::Graph& H, vertex v,
                                                                 std::optional<vertex> last_neighbour_index);
};
} // namespace pattern