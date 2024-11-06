#include "core.h"
#include "find_embedding/find_embedding.hpp"
#include "find_embedding/graph.hpp"
#include "miner_minor_matcher.hpp"
#include <vector>

class MyCppInteractions : public find_embedding::LocalInteraction {
  public:
    bool _canceled = false;
    void cancel() {
        _canceled = true;
    }

  private:
    void displayOutputImpl(int, const std::string& mess) const override {
        std::cout << mess << std::endl;
    }
    void displayErrorImpl(int, const std::string& mess) const override {
        std::cerr << mess << std::endl;
    }
    bool cancelledImpl() const override {
        return _canceled;
    }
};

namespace pattern
{
bool MinerMinorMatcher::match(const core::Graph& G, const core::Graph& Q) {
    find_embedding::optional_parameters params;
    params.localInteractionPtr.reset(new MyCppInteractions);
    auto bigGraph = this->convert_graph(G);
    auto smallGraph = this->convert_graph(Q);
    std::vector<std::vector<int>> chains;
    return find_embedding::findEmbedding(bigGraph, smallGraph, params, chains);
}

graph::input_graph MinerMinorMatcher::convert_graph(const core::Graph& G) {

    auto edges = G.edges();
    auto edges_start = std::vector<int>(G.edges().size());
    auto edges_end = std::vector<int>(edges_start.size());

    int index = 0;
    for (auto [u, v] : edges) {
        edges_start[index] = u;
        edges_end[index] = v;
        index++;
    }
    graph::input_graph convertedGraph = graph::input_graph(G.size(), edges_start, edges_end);
    return convertedGraph;
}
} // namespace pattern
