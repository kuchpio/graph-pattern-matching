#pragma once

#include "minor_matcher.h"

namespace pattern
{
class InducedMinorMatcher : public MinorMatcher {
  public:
    bool match(const core::Graph& G, const core::Graph& H) override;

  private:
    bool minor_induced_recursion(const core::Graph& G, const core::Graph& H, int v);
};
} // namespace pattern