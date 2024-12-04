#include "core.h"
#include "cuda_subgraph_matcher.h"
#include <optional>
#include <vector>
#include <algorithm>

#include <cub/cub.cuh>
#include <set>

// for signatures
#define SIGLEN 64 * 8
#define VLEN 32
#define SIGNUM SIGLEN / VLEN
#define SIGBYTE sizeof(unsigned) * SIGNUM
#define HASHSEED 17
#define HASHSEED2 53

namespace pattern
{
__global__ void createCandidateSetKernel(int u, bool* candidateSet, const CudaGraph& bigGraph,
                                         const CudaGraph& smallGraph) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= *bigGraph.dev_size) return;
    // check neighbours
    uint32_t uNeighboursDegrees = 0;
    uint32_t vNeighboursDegrees = 0;
    auto uNeighboursOut = smallGraph.neighboursOut(u);
    auto vNeighboursOut = bigGraph.neighboursOut(v);

    for (int i = 0; i < vNeighboursOut; i++)
        vNeighboursDegrees += bigGraph.dev_neighbours[i + bigGraph.dev_neighboursOffset[v]];
    for (int i = 0; i < uNeighboursOut; i++)
        uNeighboursDegrees += smallGraph.dev_neighbours[i + smallGraph.dev_neighboursOffset[u]];

    if (uNeighboursDegrees == vNeighboursDegrees) candidateSet[v] = true;

    // calculate signature
}

__global__ void filterCandidates(uint32_t* candidates, uint32_t* prefixScan, bool* candidateSet,
                                 uint32_t* candidatesCount, uint32_t* size) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= *size) return;
    if (candidateSet[v]) candidates[prefixScan[v]] = v;

    if (v == *size - 1) *candidatesCount = prefixScan[v];
}

CudaGraph::CudaGraph(const core::Graph& G) {
    neighbours = std::vector<uint32_t>(G.edge_count());
    neighboursOffset = std::vector<uint32_t>(G.size());

    uint32_t offset = 0;
    for (uint32_t v = 0; v < G.size(); v++) {
        neighboursOffset[v] = offset;
        for (uint32_t u : G.get_neighbours(v)) {
            neighbours[offset++] = u;
        }
    }
    this->allocGPU();
}

void CudaGraph::allocGPU() {
    this->dev_neighboursOffset = cuda::malloc<uint32_t>(this->neighboursOffset.size());
    this->dev_neighboursOffset = cuda::malloc<uint32_t>(this->neighbours.size());
    this->dev_size = cuda::malloc<uint32_t>(1);
    this->dev_edgeCount = cuda::malloc<uint32_t>(1);
    auto size = this->size();
    uint32_t edgeCount = this->neighbours.size();

    cuda::memcpy_host_dev<uint32_t>(dev_neighboursOffset, neighboursOffset.data(), neighboursOffset.size());
    cuda::memcpy_host_dev<uint32_t>(dev_neighbours, neighbours.data(), neighbours.size());
    cuda::memcpy_host_dev<uint32_t>(dev_size, &size, 1);
    cuda::memcpy_host_dev<uint32_t>(dev_edgeCount, &edgeCount, 1);
}

void CudaGraph::freeGPU() {
    cuda::free(dev_neighbours);
    cuda::free(dev_neighboursOffset);
    cuda::free(dev_size);
    cuda::free(dev_edgeCount);
}

__device__ uint32_t CudaGraph::neighboursOut(uint32_t v) const {
    if (v < (*this->dev_size - 1)) {
        return this->dev_neighboursOffset[v + 1] - this->dev_neighboursOffset[v];
    }
    return this->edgeCount() - this->dev_neighboursOffset[v];
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::match(const core::Graph& bigGraph,
                                                              const core::Graph& smallGraph) {
    if (smallGraph.size() > bigGraph.size()) return std::nullopt;
    auto processedVertices = std::set<vertex>();

    auto bigCudaGraph = CudaGraph(bigGraph);
    auto smallCudaGraph = CudaGraph(smallGraph);

    // DO IT ON GPU
    candidateLists_ = createCandidateLists(bigGraph, smallGraph);
    // SYNC CUDA

    // Process first vertex
    uint32_t firstVertex = this->getNextVertex(smallGraph, candidateLists_, processedVertices);
    processedVertices.insert(firstVertex);
    dev_result_ = cuda::malloc<uint32_t>(candidateLists_[firstVertex].size());
    cuda::memcpy_host_dev(dev_result_, candidateLists_[firstVertex].data(), candidateLists_[firstVertex].size());

    for (int v = 1; v < smallGraph.size(); v++) {
    }

    return std::nullopt;
}

std::vector<std::vector<uint32_t>> CudaSubgraphMatcher::createCandidateLists(const CudaGraph& bigGraph,
                                                                             const CudaGraph& smallGraph) {
    auto candidateLists = std::vector<std::vector<uint32_t>>(smallGraph.size());

    uint32_t num_blocks = (bigGraph.size() + block_size_ - 1) / block_size_;

    bool* dev_candidateSet = cuda::malloc<bool>(bigGraph.size());
    uint32_t* dev_prefixScan = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_candidates = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_candidateCount = cuda::malloc<uint32_t>(1);

    for (uint32_t u = 0; u < smallGraph.size(); u++) {
        cuda::memset<bool>(dev_candidateSet, 0, bigGraph.size());
        createCandidateSetKernel<<<num_blocks, block_size_>>>(u, dev_candidateSet, bigGraph, smallGraph);
        cuda::ExclusiveSum(dev_prefixScan, dev_candidateSet, bigGraph.size());
        filterCandidates<<<num_blocks, block_size_>>>(dev_candidates, dev_prefixScan, dev_candidateSet,
                                                      dev_candidateCount, bigGraph.dev_size);
        uint32_t candidateCount = 0;
        cuda::memcpy_dev_host<uint32_t>(&candidateCount, dev_candidateCount, 1);

        auto candidates = std::vector<uint32_t>(candidateCount);
        cuda::memcpy_dev_host(candidates.data(), dev_candidates, candidateCount);
        candidateLists[u] = candidates;
    }

    cuda::free(dev_candidateSet);
    cuda::free(dev_prefixScan);
    cuda::free(dev_candidates);
    cuda::free(dev_candidateCount);
    return candidateLists;
}

uint32_t CudaSubgraphMatcher::getNextVertex(const CudaGraph& graph,
                                            const std::vector<std::vector<uint32_t>>& candidateLists,
                                            const std::set<vertex>& processedVertices) {
    uint32_t highestScoreVertex = 0;
    uint32_t currentMax = 0;
    for (vertex v = 0; v < graph.size(); v++) {
        if (processedVertices.contains(v)) continue;
        if (candidateLists[v].size() / graph.neighboursOut(v) > currentMax) {
            currentMax = candidateLists[v].size() / graph.neighboursOut(v);
            highestScoreVertex = v;
        }
    }
    return highestScoreVertex;
}

} // namespace pattern