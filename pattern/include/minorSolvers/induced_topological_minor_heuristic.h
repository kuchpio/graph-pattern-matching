#pragma once

#include "minor_matcher.h"
#include "vf2_induced_subgraph_solver.hpp"
namespace pattern
{
class InducedTopologicalMinorHeuristic : public MinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  private:
    std::optional<std::vector<vertex>> tpRecursion(const core::Graph& G, const core::Graph H, int depth);
    static core::Graph subdivideEdge(const core::Graph& G, vertex u, vertex v);
    Vf2InducedSubgraphSolver subgraphSolver = Vf2InducedSubgraphSolver();
};
} // namespace pattern