#include "topological_induced_minor_matcher.h"
#include "isomorphism_matcher.h"

namespace pattern
{
bool TopologicalInducedMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return topological_induced_minor_recursion(G, H, 0, {});
};

bool TopologicalInducedMinorMatcher::topological_induced_minor_recursion(const core::Graph& G, const core::Graph& H,
                                                                         vertex v,
                                                                         std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (topological_induced_minor_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        auto G_after_edge_contraction = contract_edge(G, v, G.get_neighbours(v)[0]);
        if (topological_induced_minor_recursion(G_after_edge_contraction, H, v, std::nullopt)) return true;
    }

    return topological_induced_minor_recursion(G, H, v + 1, std::nullopt);
}

} // namespace pattern