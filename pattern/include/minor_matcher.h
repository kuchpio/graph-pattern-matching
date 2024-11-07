#pragma once

#include "core.h"
#include "isomorphism_matcher.h"
#include "pattern.h"
#include <optional>
namespace pattern
{
class MinorMatcher : public PatternMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H) override;

  protected:
    bool minor_recursion(const core::Graph& G, const core::Graph& H, std::size_t v,
                         std::optional<int> last_neighbour_index);
    static core::Graph remove_vertex(const core::Graph& G, int v);
    static core::Graph remove_edge(const core::Graph& G, int u, int v);
    static core::Graph contract_edge(const core::Graph& G, int u, int v);
    IsomorphismMatcher isomorphismMatcher = IsomorphismMatcher();
};

} // namespace pattern