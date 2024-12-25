#pragma once

#include "vf2_induced_subgraph_solver.hpp"
#include "topological_minor_heuristic.h"

namespace pattern
{
class InducedMinorHeuristic : public TopologicalMinorHeuristic {
  public:
    InducedMinorHeuristic() : TopologicalMinorHeuristic(nullptr) {};

  protected:
    std::optional<std::vector<vertex>> tpRecursion(const core::Graph G, const core::Graph& H,
                                                   const std::vector<vertex>& mapping, int depth,
                                                   int lastSkippedEdge) override;
    Vf2InducedSubgraphSolver subgraphSolver = Vf2InducedSubgraphSolver();
};
} // namespace pattern