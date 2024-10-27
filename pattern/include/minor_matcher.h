#pragma once

#include "core.h"
#include "pattern.h"
#include <optional>
namespace pattern
{
class MinorMatcher : public PatternMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H);
    bool induced_minor(const core::Graph& G, const core::Graph& H);

  protected:
    bool minor_recursion(const core::Graph& G, const core::Graph& H, int v, std::optional<int> last_neighbour_index);
    static core::Graph remove_vertex(const core::Graph& G, int v);
    static core::Graph remove_edge(const core::Graph& G, int u, int v);
    static core::Graph extract_edge(const core::Graph& G, int u, int v);
};

} // namespace pattern