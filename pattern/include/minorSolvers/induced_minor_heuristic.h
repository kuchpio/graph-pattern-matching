#pragma once

#include "vf2_induced_subgraph_solver.hpp"
#include "minor_heuristic.h"
#include <set>

namespace pattern
{
class InducedMinorHeuristic : public MinorHeuristic {
  public:
    InducedMinorHeuristic(bool directed = false)
        : MinorHeuristic(std::make_unique<Vf2InducedSubgraphSolver>(), directed){};
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& H) override;

  protected:
    std::optional<std::vector<vertex>> inducedMinorRecursion(const core::Graph& G, const core::Graph& H,
                                                             const std::vector<vertex>& mapping,
                                                             std::set<std::tuple<vertex, vertex>> processedEdges,
                                                             int depth, std::size_t lastSkippedEdge);
    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                      const std::vector<vertex>& mapping, int depth,
                                                      int lastSkippedEdge) override {
        return std::nullopt;
    }
    bool maxDegreeConstraint(const core::Graph& G, const core::Graph& H);

    std::vector<std::tuple<vertex, vertex>> edges_;
};
} // namespace pattern