#pragma once

#include "core.h"
#include "pattern.h"
#include "argraph.h"
#include <optional>

namespace pattern
{
class Vf2InducedSubgraphSolver : public PatternMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
    std::optional<std::vector<vertex>> matching(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    Graph convert_graph(const core::Graph& G);
};
} // namespace pattern