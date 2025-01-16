#pragma once

#include "vf2_induced_subgraph_matcher.hpp"
#include "minor_exact.h"
#include <set>

namespace pattern
{
class InducedMinorExactMatcher : public MinorExact {
  public:
    InducedMinorExactMatcher(bool directed = false)
        : MinorExact(std::make_unique<Vf2InducedSubgraphMatcher>(), directed){};
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override;

  protected:
    std::optional<std::vector<vertex>> inducedMinorRecursion(const core::Graph& G, const core::Graph& H,
                                                             const std::vector<vertex>& mapping,
                                                             std::set<std::tuple<vertex, vertex>> processedEdges,
                                                             int depth, std::size_t lastSkippedEdge);
    bool maxDegreeConstraint(const core::Graph& G, const core::Graph& H);

    std::vector<std::tuple<vertex, vertex>> edges_;
};
} // namespace pattern