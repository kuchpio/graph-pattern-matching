#pragma once
#include "plugin.h"

#include <optional>
#include <vector>

class SamplePatternMatcher : public core::IPatternMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& searchSpace, const core::Graph& pattern);
    void interrupt();
};
