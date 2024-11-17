#pragma once

#include "core.h"
#include "find_embedding/util.hpp"
#include "find_embedding/find_embedding.hpp"
#include "find_embedding/graph.hpp"
#include "minor_matcher.h"
namespace pattern
{
class MinerMinorMatcher : public MinorMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H) override;
    std::optional<std::vector<vertex>> matching(const core::Graph& G, const core::Graph& H);

  private:
    graph::input_graph convert_graph(const core::Graph& G);
};

} // namespace pattern