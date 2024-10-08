#include "pattern.h"
#include <map>

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return bigGraph.size() >= smallGraph.size();
}

bool is_sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph) {
    return is_isomorphism(bigGraph, smallGraph);
}

bool is_isomorphism(const core::Graph& G, const core::Graph& Q) {

    if (G.size() != Q.size()) return false;

    std::map<int, int> mapping;

    // zaczynami od pierwszego wierzczholka

    return false;
}

bool is_isomorphis_recursion(const core::Graph& G, const core::Graph& Q, std::map<int, int>& mapping) {
    // mamy juz cos przypisane
    // sprawdz ktore wierzcholki z mapowania nie maja wszystkich sasiadow, sproboj zmapowac pierwszego
}
} // namespace pattern
