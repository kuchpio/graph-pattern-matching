#pragma once

#include "core.h"
#include <optional>
#include <unordered_map>
#include <vector>
#include <atomic>

namespace pattern
{
class PatternMatcher {
  public:
    virtual ~PatternMatcher() = default;
    virtual std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph) = 0;
    void interrupt() {
        interrupted_ = true;
    }

  protected:
    std::atomic_bool interrupted_ = false;
    inline std::vector<vertex> getMatching(std::unordered_map<vertex, vertex> mapping) {
        std::vector<vertex> matching = std::vector<vertex>(mapping.size());
        for (const auto& pair : mapping) {
            matching[pair.first] = pair.second;
        }
        return matching;
    }
};
} // namespace pattern
