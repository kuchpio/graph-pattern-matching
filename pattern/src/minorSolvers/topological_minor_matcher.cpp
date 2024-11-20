#include "topological_minor_matcher.h"
#include "core.h"
#include <optional>
#include <vector>

namespace pattern
{
bool TopologicalMinorMatcher::match(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return false;

    return topological_minor_recursion(G, H, 0, std::nullopt);
};

std::optional<std::vector<vertex>> TopologicalMinorMatcher::matching(const core::Graph& G, const core::Graph& H) {
    if (H.size() > G.size()) return std::nullopt;

    return topologicalMinorRecursion(G, H, 0, std::nullopt);
};

bool TopologicalMinorMatcher::topological_minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                                                          std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return false;

    if (this->isomorphismMatcher.match(G, H)) return true;

    if (v >= G.size()) return false;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        if (topological_minor_recursion(G_after_removal, H, v + 1, std::nullopt)) return true;
    }

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        auto G_after_edge_contraction = contract_edge(G, v, G.get_neighbours(v)[0]);
        if (topological_minor_recursion(G_after_edge_contraction, H, v, std::nullopt)) return true;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        if (topological_minor_recursion(G_after_edge_removal, H, v, neighbour_index)) return true;
    }

    return topological_minor_recursion(G, H, v + 1, std::nullopt);
}

std::optional<std::vector<vertex>> TopologicalMinorMatcher::topologicalMinorRecursion(
    const core::Graph& G, const core::Graph& H, vertex v, std::optional<vertex> last_neighbour_index) {
    if (H.size() > G.size()) return std::nullopt;

    auto matching = this->isomorphismMatcher.matching(G, H);
    if (this->isomorphismMatcher.match(G, H)) return matching;

    if (v >= G.size()) return std::nullopt;

    vertex start_neighbour_index = last_neighbour_index.value_or(0);

    if (G.size() > H.size()) {
        auto G_after_removal = remove_vertex(G, v);
        auto matching = topologicalMinorRecursion(G_after_removal, H, v + 1, std::nullopt);
        if (matching) return matching;
    }

    if (G.degree_in(v) == 1 && G.degree_out(v) == 1) {
        auto G_after_edge_contraction = contract_edge(G, v, G.get_neighbours(v)[0]);
        auto matching = topologicalMinorRecursion(G_after_edge_contraction, H, v, std::nullopt);
        if (matching) return matching;
    }

    for (vertex neighbour_index = start_neighbour_index; neighbour_index < G.get_neighbours(v).size();
         neighbour_index++) {
        auto neighbour = G.get_neighbours(v)[neighbour_index];

        auto G_after_edge_removal = remove_edge(G, v, neighbour);

        auto matching = topologicalMinorRecursion(G_after_edge_removal, H, v, neighbour_index);
        if (matching) return matching;
    }

    return topologicalMinorRecursion(G, H, v + 1, std::nullopt);
}

} // namespace pattern