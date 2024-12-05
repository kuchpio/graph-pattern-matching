#include "core.h"
#include "cuda_subgraph_matcher.h"
#include <optional>
#include <vector>
#include <algorithm>

#include <cub/cub.cuh>
#include <set>

namespace pattern
{
__global__ void createCandidateSetKernel(int u, uint32_t* candidateSet, const CudaGraph& bigGraph,
                                         const CudaGraph& smallGraph) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= *bigGraph.dev_size) return;
    // check neighbours
    uint32_t uNeighboursDegrees = 0;
    uint32_t vNeighboursDegrees = 0;
    auto uNeighboursOut = smallGraph.dev_neighboursOut(u);
    auto vNeighboursOut = bigGraph.dev_neighboursOut(v);

    for (int i = 0; i < vNeighboursOut; i++)
        vNeighboursDegrees += bigGraph.dev_neighbours[i + bigGraph.dev_neighboursOffset[v]];
    for (int i = 0; i < uNeighboursOut; i++)
        uNeighboursDegrees += smallGraph.dev_neighbours[i + smallGraph.dev_neighboursOffset[u]];

    if (uNeighboursDegrees == vNeighboursDegrees) candidateSet[v] = true;

    // calculate signature
}

__global__ void filterCandidates(uint32_t* candidates, uint32_t* prefixScan, uint32_t* candidateSet, uint32_t* size) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= *size) return;
    if (candidateSet[v]) candidates[prefixScan[v]] = v;
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

__device__ uint32_t CudaGraph::dev_neighboursOut(uint32_t v) const {
    if (v < (*this->dev_size - 1)) {
        return this->dev_neighboursOffset[v + 1] - this->dev_neighboursOffset[v];
    }
    return this->edgeCount() - this->dev_neighboursOffset[v];
}

__host__ uint32_t CudaGraph::neighboursOut(uint32_t v) const {
    if (v < (this->size() - 1)) {
        return this->neighboursOffset[v + 1] - this->neighboursOffset[v];
    }
    return this->neighbours.size() - this->neighboursOffset[v];
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::match(const core::Graph& bigGraph,
                                                              const core::Graph& smallGraph) {
    if (smallGraph.size() > bigGraph.size()) return std::nullopt;

    auto bigCudaGraph = CudaGraph(bigGraph);
    auto smallCudaGraph = CudaGraph(smallGraph);

    candidatesSizes_ = createCandidateLists(bigGraph, smallGraph, dev_candidates_);

    // Process first vertex
    uint32_t firstVertex = this->getNextVertex(smallGraph, candidatesSizes_, processedVertices_);
    processedVertices_.insert(firstVertex);
    dev_result_ = dev_candidates_[firstVertex];

    for (int v = 1; v < smallGraph.size(); v++) {
        auto nextVertex = this->getNextVertex(smallGraph, candidatesSizes_, processedVertices_);
        processedVertices_.insert(nextVertex);
    }

    return std::nullopt;
}

std::vector<uint32_t> CudaSubgraphMatcher::createCandidateLists(const CudaGraph& bigGraph, const CudaGraph& smallGraph,
                                                                uint32_t** dev_candidatesList) {
    auto candidateListsSizes = std::vector<uint32_t>(smallGraph.size());

    uint32_t num_blocks = (bigGraph.size() + block_size_ - 1) / block_size_;

    uint32_t* dev_candidateSet = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_prefixScan = cuda::malloc<uint32_t>(bigGraph.size());

    uint32_t* dev_candidateCount = cuda::malloc<uint32_t>(1);

    for (uint32_t u = 0; u < smallGraph.size(); u++) {
        cuda::memset<uint32_t>(dev_candidateSet, 0, bigGraph.size());
        createCandidateSetKernel<<<num_blocks, block_size_>>>(u, dev_candidateSet, bigGraph, smallGraph);
        cuda::ExclusiveSum<uint32_t>(dev_prefixScan, dev_candidateSet, bigGraph.size());
        cuda::memcpy_dev_host<uint32_t>(&candidateListsSizes[u], &dev_prefixScan[bigGraph.size() - 1], 1);

        uint32_t* dev_candidates = cuda::malloc<uint32_t>(candidateListsSizes[u]);
        filterCandidates<<<num_blocks, block_size_>>>(dev_candidates, dev_prefixScan, dev_candidateSet,
                                                      &candidateListsSizes[u]);
        dev_candidatesList[u] = dev_candidates;
    }

    cuda::free(dev_candidateSet);
    cuda::free(dev_prefixScan);
    // cuda::free(dev_candidates);
    cuda::free(dev_candidateCount);
    return candidateListsSizes;
}

uint32_t CudaSubgraphMatcher::getNextVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidateListsSizes,
                                            const std::set<uint32_t>& processedVertices) {
    uint32_t highestScoreVertex = 0;
    uint32_t currentMax = 0;
    for (uint32_t v = 0; v < graph.size(); v++) {
        if (processedVertices.contains(v)) continue;
        if (candidateListsSizes[v] / graph.neighboursOut(v) > currentMax) {
            currentMax = candidateListsSizes[v] / graph.neighboursOut(v);
            highestScoreVertex = v;
        }
    }
    return highestScoreVertex;
}

void addVertexToResultTable(int v, const std::vector<std::vector<uint32_t>>& candidateLists_,
                            const core::Graph& Graph) {
}

} // namespace pattern