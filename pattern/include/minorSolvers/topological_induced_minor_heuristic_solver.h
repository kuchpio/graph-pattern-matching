#pragma once

#include "topological_minor_heuristic.h"
#include "vf2_induced_subgraph_solver.hpp"

namespace pattern
{
class InducedTopologicalMinorHeuristicSolver : public TopologicalMinorHeuristic {
  public:
    InducedTopologicalMinorHeuristicSolver(bool direct = false) : TopologicalMinorHeuristic(nullptr, direct) {
        subgraphMatcher_ = std::make_unique<Vf2InducedSubgraphSolver>();
    };
};
} // namespace pattern