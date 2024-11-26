#include "induced_minor_matcher.h"
#include "core.h"
#include <optional>
#include <vector>

namespace pattern
{
std::optional<std::vector<vertex>> InducedMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;

    return minorInducedRecursion(G, H, 0, std::nullopt);
}

bool InducedMinorMatcher::minor_induced_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                                  std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (minor_induced_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];
        auto G_after_edge_contraction = contract_edge(G, v, neighbour);
        if (minor_induced_recursion(G_after_edge_contraction, H, v, neighbour_index)) return true;
    }

    return minor_induced_recursion(G, H, v + 1, std::nullopt);
}

std::optional<std::vector<vertex>> InducedMinorMatcher::minorInducedRecursion(
    const core::Graph& G, const core::Graph& H, vertex v, std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return std::nullopt;

    auto match = this->isomorphismMatcher.match(G, H);
    if (match) return match;

    if (v >= G.size()) return std::nullopt;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        auto match = minorInducedRecursion(G_after_removal, H, v + 1, std::nullopt);
        if (match) return match;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];
        auto G_after_edge_contraction = contract_edge(G, v, neighbour);
        auto match = minorInducedRecursion(G_after_edge_contraction, H, v, neighbour_index);
        if (match) return match;
    }

    return minorInducedRecursion(G, H, v + 1, std::nullopt);
}
} // namespace pattern