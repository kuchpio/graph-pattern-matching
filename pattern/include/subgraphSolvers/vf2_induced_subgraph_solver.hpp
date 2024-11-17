#pragma once

#include "core.h"
#include "pattern.h"
#include "argraph.h"
#include "subgraph_matcher.h"
#include <optional>
#include <tuple>

namespace pattern
{
class Vf2InducedSubgraphSolver : public SubgraphMatcher {
  public:
    bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
    std::optional<std::vector<vertex>> matching(const core::Graph& bigGraph, const core::Graph& smallGraph);

  private:
    Graph convert_graph(const core::Graph& G);
};
} // namespace pattern