#pragma once

#include "vf2_induced_subgraph_solver.hpp"
#include "minor_heuristic.h"

namespace pattern
{
class InducedMinorHeuristic : public MinorHeuristic {
  public:
    InducedMinorHeuristic() : MinorHeuristic(std::make_unique<Vf2InducedSubgraphSolver>()){};

  protected:
    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H,
                                                      const std::vector<vertex>& mapping, int depth,
                                                      int lastSkippedEdge) override;
    bool maxDegreeConstraint(const core::Graph& G, const core::Graph& H);
};
} // namespace pattern