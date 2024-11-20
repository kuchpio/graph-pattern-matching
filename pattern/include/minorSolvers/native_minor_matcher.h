#pragma once

#include "core.h"
#include "minor_matcher.h"
#include "vf2_isomorphism_solver.hpp"
#include <optional>
#include <vector>
namespace pattern
{
class NativeMinorMatcher : public MinorMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& G, const core::Graph& Q) override;

  protected:
    bool minor_recursion(const core::Graph& G, const core::Graph& H, vertex v,
                         std::optional<vertex> last_neighbour_index);

    std::optional<std::vector<vertex>> minorRecursion(const core::Graph& G, const core::Graph& H, vertex v,
                                                      std::optional<vertex> last_neighbour_index);
    static core::Graph remove_vertex(const core::Graph& G, vertex v);
    static core::Graph remove_edge(const core::Graph& G, vertex u, vertex v);
    static core::Graph contract_edge(const core::Graph& G, vertex u, vertex v);
    Vf2IsomorphismSolver isomorphismMatcher = Vf2IsomorphismSolver();
};

} // namespace pattern