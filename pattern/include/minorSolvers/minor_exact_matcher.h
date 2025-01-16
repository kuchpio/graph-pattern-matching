#pragma once
#pragma once

#include "vf2_subgraph_matcher.hpp"
#include "minor_exact.h"
#include <set>

namespace pattern
{
class MinorExactMatcher : public MinorExact {
  public:
    MinorExactMatcher(bool directed = false) : MinorExact(std::make_unique<Vf2SubgraphMatcher>(), directed){};
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override;

  protected:
    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                      const std::vector<vertex>& mapping,
                                                      std::set<std::tuple<vertex, vertex>> processedEdges, int depth,
                                                      std::size_t lastSkippedEdge);

    std::vector<std::tuple<vertex, vertex>> edges_;
};
} // namespace pattern