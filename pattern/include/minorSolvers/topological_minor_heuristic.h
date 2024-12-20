#pragma once

#include "minor_matcher.h"
#include "subgraph_matcher.h"

namespace pattern
{
class TopologicalMinorHeuristic : public MinorMatcher {
  public:
    TopologicalMinorHeuristic(SubgraphMatcher* subgraphMatcher) : subgraphMatcher_(subgraphMatcher){};
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  protected:
    std::optional<std::vector<vertex>> tpRecursion(const core::Graph G, const core::Graph& H, int depth);
    static core::Graph contractEdge(const core::Graph& G, vertex u, vertex v);
    SubgraphMatcher* subgraphMatcher_;
};
} // namespace pattern