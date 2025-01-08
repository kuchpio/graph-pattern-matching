#pragma once
#include <string>
#include "core.h"
#include <CLI/CLI.hpp>

struct CLIGraph {
    std::string filepath;
    std::size_t index;
    core::Graph graph;
};

class GraphPatternMatchingCLI {
  public:
    GraphPatternMatchingCLI();
    void parse(int argc, char** argv);
    int exit(const CLI::ParseError& error);
    void run() const;

  private:
    CLI::App app_{"Graph Pattern Matching"};
    bool induced = false;
};
