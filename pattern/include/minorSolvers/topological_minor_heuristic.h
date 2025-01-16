#pragma once

#include "minor_heuristic.h"

namespace pattern
{
class TopologicalMinorHeuristic : public MinorHeuristic {
  public:
    TopologicalMinorHeuristic(std::unique_ptr<SubgraphMatcher> subgraphMatcher, bool directed = false)
        : MinorHeuristic(std::move(subgraphMatcher), directed) {
        if (directed) maxDeegre_ = 1;
    };

  protected:
    std::size_t maxDeegre_ = 2;

    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                      const std::vector<vertex>& mapping, int depth,
                                                      int lastSkippedEdge) override;
};

} // namespace pattern