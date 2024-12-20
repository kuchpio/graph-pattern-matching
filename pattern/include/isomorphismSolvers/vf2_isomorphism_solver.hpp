#pragma once

#include "core.h"
#include "argraph.h"
#include "subgraph_matcher.h"
#include <optional>
#include "vf2solver.h"
#include "vf2_state.h"

namespace pattern
{
class Vf2IsomorphismSolver : public SubgraphMatcher, public Vf2Solver {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
        auto G = convertGraph(bigGraph);
        auto Q = convertGraph(smallGraph);
        VF2State s0(&Q, &G);

        return processMatching(s0, bigGraph, smallGraph);
    }
};
} // namespace pattern