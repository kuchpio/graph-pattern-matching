#pragma once

#include "core.h"
#include "utils.h"

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool sub_edge_induced_isomporhism(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool sub_induced_isomorpshim(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool connected_isomorphism(const core::Graph& G, const core::Graph& Q);
bool isomorphism(const core::Graph& G, const core::Graph& Q);
} // namespace pattern
