#pragma once

#include "core.h"
#include "native_minor_matcher.h"
#include <optional>
#include <vector>

namespace pattern
{
class InducedMinorMatcher : public NativeMinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  private:
    bool minor_induced_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                 std::optional<vertex> last_neighbour_index);
    std::optional<std::vector<vertex>> minorInducedRecursion(const core::Graph& G, const core::Graph& H, vertex v,
                                                             std::optional<vertex> last_neighbour_index);
};
} // namespace pattern