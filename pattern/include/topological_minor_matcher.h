#pragma once
#include "core.h"
#include "minor_matcher.h"
namespace pattern
{
class TopologicalMinorMatcher : public MinorMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H);

  private:
    bool topological_minor_recursion(const core::Graph& G, const core::Graph& H, int v,
                                     std::optional<int> last_neighbour_index);
};
} // namespace pattern