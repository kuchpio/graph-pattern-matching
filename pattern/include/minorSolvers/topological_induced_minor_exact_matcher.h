#pragma once

#include "topological_minor_exact.h"
#include "vf2_induced_subgraph_matcher.hpp"

namespace pattern
{
class InducedTopologicalMinorExactMatcher : public TopologicalMinorExact {
  public:
    InducedTopologicalMinorExactMatcher(bool direct = false) : TopologicalMinorExact(nullptr, direct) {
        subgraphMatcher_ = std::make_unique<Vf2InducedSubgraphMatcher>();
    };
};
} // namespace pattern