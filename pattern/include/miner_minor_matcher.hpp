#pragma once

#include "core.h"
#include "find_embedding/util.hpp"
#include "isomorphism_matcher.h"
#include "find_embedding/find_embedding.hpp"
#include "find_embedding/graph.hpp"

namespace pattern
{
class MinerMinorMatcher : public PatternMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H) override;

  private:
    graph::input_graph convert_graph(const core::Graph& G);
};

} // namespace pattern