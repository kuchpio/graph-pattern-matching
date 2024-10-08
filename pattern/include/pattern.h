#pragma once

#include "core.h"

namespace pattern
{
bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool is_sub_isomorphism(const core::Graph& bigGraph, const core::Graph& smallGraph);
bool is_isomorphism(const core::Graph& G, const core::Graph& Q);
} // namespace pattern
