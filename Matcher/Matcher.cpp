#include "Matcher.h"

namespace mtr 
{
	bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
		return bigGraph.size() >= smallGraph.size();
	}
}
