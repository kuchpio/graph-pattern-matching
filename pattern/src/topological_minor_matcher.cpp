#include "topological_minor_matcher.h"
#include "isomorphism_matcher.h"
#include <cstddef>
#include <optional>

namespace pattern
{
bool TopologicalMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return topological_minor_recursion(G, H, 0, std::nullopt);
};

bool TopologicalMinorMatcher::topological_minor_recursion(const core::Graph& G, const core::Graph& H, int v,
                                                          std::optional<int> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    int start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (topological_minor_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        auto G_after_edge_extraction = extract_edge(G, v, G.get_neighbours(v)[0]);
        if (topological_minor_recursion(G_after_edge_extraction, H, v, std::nullopt)) return true;
    }

    for (int neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size(); neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        if (topological_minor_recursion(G_after_edge_removal, H, v, neighbour_index)) return true;
    }

    return topological_minor_recursion(G, H, v + 1, std::nullopt);
}

/*
bool TopologicalMinorMatcher::topological_minor_recursion(const core::Graph& G, const core::Graph& H, int v,
                                                          std::optional<int> last_neighbour_index) {

    if (H.size() > G.size()) return false;

    auto isomorphismMatcher = IsomorphismMatcher();
    if (isomorphismMatcher.match(G, H)) return true;

    if (v > G.size()) return false;

    int start_neighbour_index = last_neighbour_index.value_or(0);
    if (topological_minor_recursion(remove_vertex(G, v), H, v, {})) return true;

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        if (topological_minor_recursion(extract_edge(G, v, G.get_neighbours(v)[0]), H, v, {})) return true;
    }
    for (int neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size(); neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];
        if (topological_minor_recursion(remove_edge(G, v, neighbour), H, v, neighbour_index)) return true;
    }
    return topological_minor_recursion(G, H, v + 1, {});
}
*/

} // namespace pattern