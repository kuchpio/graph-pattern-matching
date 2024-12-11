#ifndef CUDA_SUBGRAPH_MATCHER_H
#define CUDA_SUBGRAPH_MATCHER_H

#include "core.h"
#include "subgraph_matcher.h"
#include "cuda_runtime.h"

#include <vector>
#include <cstdint>

namespace pattern
{

class ResultTable {
  public:
    ResultTable() : ResultTable(0) {
    }
    ResultTable(uint32_t graphSize) : dev_data(nullptr), mapping(std::vector<uint32_t>(graphSize, UINT32_MAX)) {
    }
    uint32_t* dev_data;
    uint32_t size = 0;
    uint32_t rowCount = 0;
    std::vector<uint32_t> mapping;
    void map(uint32_t v) {
        mapping[v] = size++;
    }
    void print();
};

class CudaGraph {
  public:
    CudaGraph(const core::Graph& G);

    __device__ uint32_t edgeCount() const {
        return *dev_edgeCount;
    }

    __host__ uint32_t size() const {
        return neighboursOffset.size() - 1;
    }
    __device__ uint32_t dev_neighboursOut(uint32_t v) const;
    __host__ uint32_t neighboursOut(uint32_t v) const;
    __host__ bool hasEdge(uint32_t u, uint32_t v) const {
        for (auto i = neighboursOffset[u]; i < neighboursOffset[u + 1]; i++) {
            if (neighbours[i] == v) return true;
        }
        return false;
    }

    uint32_t* dev_neighboursOffset;
    uint32_t* dev_neighbours;
    uint32_t* dev_size;
    uint32_t* dev_edgeCount;
    std::vector<uint32_t> neighboursOffset;
    std::vector<uint32_t> neighbours;
    ~CudaGraph() {
        freeGPU();
    };

  private:
    void allocGPU();
    void freeGPU();
};

class CudaSubgraphMatcher : public SubgraphMatcher {
  public:
    std::optional<std::vector<vertex>> match(const core::Graph& bigGraph, const core::Graph& smallGraph);
    static constexpr uint32_t kDefaultBlockSize = 512;

  private:
    std::vector<uint32_t> calculateScores(const core::Graph& smallGraph,
                                          const std::vector<std::vector<uint32_t>>& candidateLists);

    __host__ std::optional<std::vector<uint32_t>> createCandidateLists(const CudaGraph& bigGraph,
                                                                       const CudaGraph& smallGraph,
                                                                       std::vector<uint32_t*>& candidates);
    std::optional<uint32_t> getNextVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidatesSizes,
                                          const std::vector<uint32_t>& mapping);
    uint32_t getFirstVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidatesSizes);
    std::optional<ResultTable> addVertexToResultTable(uint32_t v, uint32_t* dev_candidates, uint32_t vCandidatesCount,
                                                      const CudaGraph& bigGraph, const CudaGraph& smallGraph,
                                                      ResultTable& resultTable);

    std::vector<uint32_t> getMappedNeighboursIn(int v, const CudaGraph& graph, const std::vector<uint32_t>& mapping);
    std::vector<uint32_t> allocateMemoryForJoining(int v, uint32_t*& GBA, const ResultTable& resultTable,
                                                   const CudaGraph& bigGraph);
    std::optional<ResultTable> linkGBAWithResult(uint32_t* dev_GBA, const std::vector<uint32_t>& GBAOffsets,
                                                 uint32_t* dev_GBAOffsets, ResultTable& resultTable,
                                                 const CudaGraph& graph, uint32_t baseIndex);

    std::optional<std::vector<vertex>> obtainResult(const ResultTable& resultTable);

    // ResultTable resultTable_ = ResultTable();
    uint32_t block_size_ = kDefaultBlockSize;
    uint32_t joiningBlockSize_ = 32;
};
} // namespace pattern

#endif