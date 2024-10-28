#include "topological_induced_minor_matcher.h"
#include "isomorphism_matcher.h"

namespace pattern
{
bool TopologicalInducedMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return topological_induced_minor_recursion(G, H, 0, {});
};

bool TopologicalInducedMinorMatcher::topological_induced_minor_recursion(const core::Graph& G, const core::Graph& H,
                                                                         int v,
                                                                         std::optional<int> last_neighbour_index) {

    if (H.size() > G.size()) return false;

    auto isomorphismMatcher = IsomorphismMatcher();
    if (isomorphismMatcher.match(G, H)) return true;

    if (v > G.size()) return false;

    int start_neighbour_index = last_neighbour_index.value_or(0);
    if (topological_induced_minor_recursion(remove_vertex(G, v), H, v, {})) return true;

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        if (topological_induced_minor_recursion(extract_edge(G, v, G.get_neighbours(v)[0]), H, v, {})) return true;
    }

    return topological_induced_minor_recursion(G, H, v + 1, {});
}

} // namespace pattern