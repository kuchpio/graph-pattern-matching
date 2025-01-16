#pragma once

#include "topological_minor_exact.h"
#include "vf2_subgraph_matcher.hpp"

namespace pattern
{
class TopologicalMinorExactMatcher : public TopologicalMinorExact {
  public:
    TopologicalMinorExactMatcher(bool direct = false) : TopologicalMinorExact(nullptr, direct) {
        subgraphMatcher_ = std::make_unique<Vf2SubgraphMatcher>();
    };
};
} // namespace pattern