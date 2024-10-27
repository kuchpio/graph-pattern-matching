#include "induced_minor_matcher.h"
#include "isomorphism_matcher.h"

namespace pattern
{

bool InducedMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return minor_induced_recursion(G, H, 0);
}

bool InducedMinorMatcher::minor_induced_recursion(const core::Graph& G, const core::Graph& H, int v) {
    auto isomorphismMatcher = IsomorphismMatcher();
    if (isomorphismMatcher.match(G, H)) return true;

    if (v > G.size()) return false;

    if (minor_induced_recursion(remove_vertex(G, v), H, v + 1)) return true;
    for (auto neighbour : G.get_neighbours(v)) {
        if (minor_induced_recursion(extract_edge(G, v, neighbour), H, v + 1)) return true;
    }
    return minor_induced_recursion(G, H, v + 1);
}
} // namespace pattern