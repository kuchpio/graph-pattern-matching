#include "core.h"
#include "subgraph_matcher.h"
#include "cuda_helpers.cuh"
#include <vector>
#include <cstdint>

namespace pattern
{

class ResultTable {
  public:
    ResultTable(uint32_t graphSize) : dev_data(nullptr), mapping(std::vector<uint32_t>(graphSize, UINT32_MAX)) {
    }
    uint32_t* dev_data;
    uint32_t size = 0;
    uint32_t rowCount = 0;
    std::vector<uint32_t> mapping;
    void map(uint32_t v) {
        mapping[v] = size++;
    }
};

class CudaGraph {
  public:
    CudaGraph(const core::Graph& G);

    __device__ uint32_t edgeCount() const {
        return *dev_edgeCount;
    }

    __host__ uint32_t size() const {
        return neighboursOffset.size();
    }
    __device__ uint32_t dev_neighboursOut(uint32_t v) const;
    __host__ uint32_t neighboursOut(uint32_t v) const;

    uint32_t* dev_neighboursOffset;
    uint32_t* dev_neighbours;
    uint32_t* dev_size;
    uint32_t* dev_edgeCount;

  private:
    void allocGPU();
    void freeGPU();

    std::vector<uint32_t> neighboursOffset;
    std::vector<uint32_t> neighbours;
};

class CudaSubgraphMatcher : SubgraphMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);
    static constexpr uint32_t kDefaultBlockSize = 512;

  private:
    std::vector<uint32_t> calculateScores(const core::Graph& smallGraph,
                                          const std::vector<std::vector<uint32_t>>& candidateLists);

    __host__ std::vector<uint32_t> createCandidateLists(const CudaGraph& bigGraph, const CudaGraph& smallGraph,
                                                        uint32_t** candidates);
    uint32_t getNextVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidatesSizes_);

    void addVertexToResultTable(int v, uint32_t* dev_candidates, const core::Graph& graph);

    std::vector<uint32_t> getMappedNeighboursIn(int v, const CudaGraph& graph);
    std::vector<uint32_t> allocateMemoryForJoining(int v, uint32_t*& GBA, ResultTable resultTable, CudaGraph graph);

    uint32_t** dev_candidates_;
    ResultTable resultTable_;
    std::vector<uint32_t> candidatesSizes_;
    uint32_t block_size_ = kDefaultBlockSize;
};
} // namespace pattern