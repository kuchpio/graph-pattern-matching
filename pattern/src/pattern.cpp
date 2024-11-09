#include "pattern.h"

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}
} // namespace pattern
