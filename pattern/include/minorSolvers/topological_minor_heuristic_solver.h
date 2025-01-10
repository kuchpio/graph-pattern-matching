#pragma once

#include "topological_minor_heuristic.h"
#include "vf2_subgraph_solver.hpp"

namespace pattern
{
class TopologicalMinorHeuristicSolver : public TopologicalMinorHeuristic {
  public:
    TopologicalMinorHeuristicSolver(bool direct = false) : TopologicalMinorHeuristic(nullptr, direct) {
        subgraphMatcher_ = std::make_unique<Vf2SubgraphSolver>();
    };
};
} // namespace pattern