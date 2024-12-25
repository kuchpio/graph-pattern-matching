#pragma once

#include "minor_matcher.h"
#include "subgraph_matcher.h"

namespace pattern
{
class TopologicalMinorHeuristic : public MinorMatcher {
  public:
    TopologicalMinorHeuristic(SubgraphMatcher* subgraphMatcher) : subgraphMatcher_(subgraphMatcher) {};
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  protected:
    virtual std::optional<std::vector<vertex>> tpRecursion(const core::Graph G, const core::Graph& H,
                                                           const std::vector<vertex>& mapping, int depth,
                                                           int lastSkippedEdge);
    static core::Graph contractEdge(const core::Graph& G, vertex u, vertex v);

    static std::vector<vertex> updateMapping(const std::vector<vertex>& mapping, vertex u, vertex v);
    static std::vector<vertex> getResult(const std::vector<vertex>& mapping,
                                         const std::vector<vertex>& contractedResult);
    SubgraphMatcher* subgraphMatcher_;
};
} // namespace pattern