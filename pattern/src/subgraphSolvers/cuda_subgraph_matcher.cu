#include "core.h"
#include "cuda_subgraph_matcher.h"
#include <optional>

namespace pattern
{
std::optional<std::vector<vertex>> CudaSubgraphMatcher::match(const core::Graph& bigGraph,
                                                              const core::Graph& smallGraph) {
    return std::nullopt;
}
} // namespace pattern