#include "induced_minor_matcher.h"
#include "isomorphism_matcher.h"

namespace pattern
{

bool InducedMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return minor_induced_recursion(G, H, 0, {});
}

bool InducedMinorMatcher::minor_induced_recursion(const core::Graph& G, const core::Graph& H, int v,
                                                  std::optional<int> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    auto isomorphismMatcher = IsomorphismMatcher();
    if (isomorphismMatcher.match(G, H)) return true;

    if (v > G.size()) return false;

    int start_neighbour_index = last_neighbour_index.value_or(0);
    if (minor_induced_recursion(remove_vertex(G, v), H, v, {})) return true;
    for (int neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size(); neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];
        if (minor_induced_recursion(remove_edge(G, v, neighbour), H, v, neighbour_index)) return true;
        if (minor_induced_recursion(extract_edge(G, v, neighbour), H, v, neighbour_index)) return true;
    }
    return minor_induced_recursion(G, H, v + 1, {});
}
} // namespace pattern