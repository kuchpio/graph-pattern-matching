#pragma once

#include "core.h"
#include "argraph.h"
#include "subgraph_matcher.h"
#include <optional>

namespace pattern
{
class Vf2SubgraphSolver : public SubgraphMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    Graph convert_graph(const core::Graph& G);
};
} // namespace pattern