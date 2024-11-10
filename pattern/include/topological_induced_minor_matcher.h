#pragma once
#include "core.h"
#include "minor_matcher.h"
namespace pattern
{
class TopologicalInducedMinorMatcher : public MinorMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H);

  private:
    bool topological_induced_minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                             std::optional<vertex> last_neighbour_index);
};
} // namespace pattern