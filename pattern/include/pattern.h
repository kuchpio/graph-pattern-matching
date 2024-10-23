#pragma once

#include "core.h"
#include "utils.h"

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool sub_induced_isomorpshim(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool connected_isomorphism(const core::Graph& G, const core::Graph& Q);
bool isomorphism(const core::Graph& G, const core::Graph& Q);
bool minor(const core::Graph& G, const core::Graph& H);
bool induced_minor(const core::Graph& G, const core::Graph& H);
bool topological_minor(const core::Graph& G, const core::Graph& H);
bool induced_topological_minor(const core::Graph& G, const core::Graph& H);
} // namespace pattern
