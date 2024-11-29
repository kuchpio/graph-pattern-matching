#include "subgraph_matcher.h"
namespace pattern
{
class CudaSubgraphMatcher : SubgraphMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);
};
} // namespace pattern