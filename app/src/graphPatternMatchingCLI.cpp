#include "graphPatternMatchingCLI.h"

GraphPatternMatchingCLI::GraphPatternMatchingCLI() {
    app_.description("CLI tool for matching patterns in graphs.");
    addMainCommandOptions();
    app_.add_subcommand("benchmarks", "run benchmarks for the algorithms.");

    app_.footer("Example:\n"
                "  ./graphPatternMatching subgraph file1 file2 \n"
                "  ./graphPatternMatching minor --induced file1 file2\n");
}

void GraphPatternMatchingCLI::addMainCommandOptions() {
    auto cmd = app_.add_subcommand("", "Find matching between graphs.");
    auto mapNameValidator = [&, this](const std::string& name) {
        if (matchingAlgorithms_.contains(name)) return std::string{};
        return std::string{"The value" + name + "is not a valid pattern."};
    };

    cmd->add_option("pattern", pattern_, "kind of pattern to be searched for.")->check(mapNameValidator);

    cmd->add_flag("-i,--induced", induced_, "makes the pattern induced.");

    cmd->add_option("searchSpaceFilePath", input1_, "path to the search space graph file")->check(CLI::ExistingFile);

    cmd->add_option("patternFilePath", input2_, "path to the pattern graph file")->check(CLI::ExistingFile);
}

void GraphPatternMatchingCLI::parse(int argc, char** argv) {
    app_.parse(argc, argv);
}

int GraphPatternMatchingCLI::exit(const CLI::ParseError& error) {
    return app_.exit(error);
}

void GraphPatternMatchingCLI::run() const {
    try {
        if (app_.got_subcommand("benchmarks")) {
            utils::EfficiencyTests benchmarks;
            benchmarks.run();
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    auto searchGraph = loadGraph(input1_);
    auto patternGraph = loadGraph(input2_);

    std::shared_ptr<pattern::PatternMatcher> matcher;

    if (induced_)
        matcher = inducedMatchingAlgorithms_.at(pattern_);
    else
        matcher = matchingAlgorithms_.at(pattern_);

    auto matching = matcher.get()->match(searchGraph, patternGraph);

    if (matching) {
        printMatching(matching.value());
        return;
    }
    std::cout << "pattern Graph is not a " + pattern_ + " of searchGraph\n";
}

core::Graph GraphPatternMatchingCLI::loadGraph(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    std::string graphEncoding;
    if (!std::getline(file, graphEncoding) || graphEncoding.empty()) {
        throw std::runtime_error("Invalid input format: missing graph");
    }

    try {
        auto graph = core::Graph6Serializer::Deserialize(graphEncoding);
        return graph;
    } catch (const core::graph6FormatError& err) {
        throw std::runtime_error("Could not open file " + filepath + "\nError: " + err.what());
    }
}

void GraphPatternMatchingCLI::printMatching(const std::vector<vertex>& matching) {
    std::cout << "Found matching: \n";
    std::cout << "[ ";
    for (auto item : matching) {
        if (item != SIZE_MAX)
            std::cout << item << " ";
        else
            std::cout << "- ";
    }
    std::cout << "]\n";
}