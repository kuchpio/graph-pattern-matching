﻿#pragma once

#include "core.h"

namespace pattern
{
class PatternMatcher {
  public:
    virtual ~PatternMatcher() = default;
    virtual bool match(const core::Graph& bigGraph, const core::Graph& smallGraph) = 0;
};

bool match(const core::Graph& bigGraph, const core::Graph& smallGraph);

} // namespace pattern
