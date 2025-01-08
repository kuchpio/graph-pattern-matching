#include "graphPatternMatchingCLI.h"

GraphPatternMatchingCLI::GraphPatternMatchingCLI() {
    app_.description("CLI tool for matching patterns in graphs.");
}

void GraphPatternMatchingCLI::parse(int argc, char** argv) {
    app_.parse(argc, argv);
}

int GraphPatternMatchingCLI::exit(const CLI::ParseError& error) {
    return app_.exit(error);
}

void GraphPatternMatchingCLI::run() const {
}