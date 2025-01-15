#include "plugin.h"
#include "sample.h"

core::IPatternMatcher* GetPatternMatcher() {
    return new SamplePatternMatcher();
}

std::optional<std::vector<vertex>> SamplePatternMatcher::match(const core::Graph& searchSpace,
                                                               const core::Graph& pattern) {
    if (pattern.size() == 0) return std::nullopt;
    std::vector<vertex> mapping(searchSpace.size(), 0);
    return mapping;
}

void SamplePatternMatcher::interrupt() {
}
