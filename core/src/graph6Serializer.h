#pragma once

#include <string>
#include <optional>
#include <stdexcept>

#include "core.h"

class graph6FormatError : public std::runtime_error {
  public:
    graph6FormatError(const std::string& message);
};

class graph6InvalidCharacterError : public graph6FormatError {
    char _message[32];
  public:
    const char* what() const override;
    graph6InvalidCharacterError(std::size_t at);
};

class Graph6Serializer {
  public:
    static core::Graph Deserialize(const std::string& graph6);
    static std::string Serialize(const core::Graph& graph);
};
