#pragma once

#include "core.h"
#include "argraph.h"
#include "subgraph_matcher.h"
#include <optional>
#include "vf2solver.h"
#include "vf2_mono_state.h"

namespace pattern
{
class Vf2SubgraphSolver : public SubgraphMatcher, public Vf2Solver {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
        auto G = convertGraph(bigGraph);
        auto Q = convertGraph(smallGraph);
        VF2MonoState s0(&Q, &G);
        s0.setInterrupted(&this->interrupted_);

        return processMatching(s0, bigGraph, smallGraph);
    }
};
} // namespace pattern