#include "core.h"
#include "cuda_subgraph_matcher.h"
#include "cuda_helpers.cuh"

#include <optional>
#include <vector>
#include <algorithm>

#include <set>

namespace pattern
{
__device__ uint32_t binarySearch(const uint32_t* arr, uint32_t size, uint32_t target);

__global__ void linkingKernel(uint32_t* dst, const ResultTable& resultTable, uint32_t* GBAPreffixSum, uint32_t* GBA,
                              uint32_t* GBAOffsets, uint32_t* neighbours) {
    uint32_t row = blockDim.x * blockIdx.x;
    uint32_t index = threadIdx.x;
    if (row > resultTable.rowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];

    uint32_t startDstIndex = GBAPreffixSum[GBAOffsets[row]];
    while (index < bufSize) {
        if (buf[index] == 1) {
            // copy
            for (uint32_t i = 0; i < resultTable.size; i++) {
                dst[startDstIndex * (resultTable.size + 1) + i] = resultTable.dev_data[row * resultTable.size + i];
            }
            dst[startDstIndex * (resultTable.size + 1) + resultTable.size] = neighbours[index];
            startDstIndex++;
        }
    };
}

__global__ void createCandidateSetKernel(uint32_t u, uint32_t* candidateSet, uint32_t bigGraphSize,
                                         uint32_t uNeighboursOut, uint32_t* bigGraphNeighbours,
                                         uint32_t* bigGraphOffsets, uint32_t* smallGraphNeighbours,
                                         uint32_t* smallGraphOffsets) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;

    if (v >= bigGraphSize) return;
    // check neighbours
    uint32_t uNeighboursDegrees = 0;
    uint32_t vNeighboursDegrees = 0;
    auto vNeighboursOut = bigGraphOffsets[v + 1] - bigGraphOffsets[v];

    for (int i = 0; i < vNeighboursOut; i++) {
        auto neighbour = bigGraphNeighbours[i + bigGraphOffsets[v]];
        vNeighboursDegrees += bigGraphOffsets[neighbour + 1] - bigGraphOffsets[neighbour];
    }
    for (int i = 0; i < uNeighboursOut; i++) {
        auto neighbour = smallGraphNeighbours[i + smallGraphOffsets[v]];
        uNeighboursDegrees += smallGraphOffsets[neighbour + 1] - smallGraphOffsets[neighbour];
    }

    if (uNeighboursDegrees == vNeighboursDegrees) candidateSet[v] = 1;
}

__global__ void filterCandidates(uint32_t* candidates, uint32_t* prefixScan, uint32_t* candidateSet, uint32_t* size) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= *size) return;
    if (candidateSet[v]) candidates[prefixScan[v]] = v;
}

__global__ void joinResultTableRowFirst(uint32_t rowCount, const ResultTable& resultTable, uint32_t* GBA,
                                        uint32_t* GBAOffsets, uint32_t* neighbours, uint32_t* candidates,
                                        uint32_t candidatesSize) {
    uint32_t row = blockDim.x * blockIdx.x;
    uint32_t index = threadIdx.x;

    if (row >= rowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];

    auto mOffset = row * resultTable.size;

    extern __shared__ uint32_t sharedRow[];

    // use shared memory here :)

    // do N(v) - m_i
    while (index < bufSize) {
        uint32_t mIndex = mOffset + index;
        while (mIndex < resultTable.size) {
            if (resultTable.dev_data[mIndex] == neighbours[index]) buf[index] = 0;
            mIndex *= 2;
        }
        index *= 2;
    }
    __syncthreads(); // commentable see if speeds up!

    index = threadIdx.x;
    // buf(i) & C(v)
    while (index < bufSize) {
        if (buf[index] == 0) continue;
        if (binarySearch(candidates, candidatesSize, neighbours[index]) == UINT32_MAX) {
            buf[index] = 0;
        }
        index *= 2;
    }
}

__global__ void joinResultTableRowSecond(uint32_t rowCount, uint32_t* GBA, uint32_t* GBAOffsets,
                                         uint32_t* currentNeighbours, uint32_t currentNeighboursSize,
                                         uint32_t* baseNeighbours) {
    uint32_t row = blockDim.x * blockIdx.x;
    uint32_t index = threadIdx.x;

    if (row >= rowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];

    extern __shared__ uint32_t sharedRow[];
    // buf(i) & N(v)
    while (index < bufSize) {
        if (buf[index] == 0) continue;
        if (binarySearch(currentNeighbours, currentNeighboursSize, baseNeighbours[index]) == UINT32_MAX) {
            buf[index] = 0;
        }
        index *= 2;
    }
}

__device__ uint32_t binarySearch(const uint32_t* arr, uint32_t size, uint32_t target) {
    uint32_t left = 0, right = size - 1;

    while (left <= right) {
        uint32_t mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return UINT32_MAX;
}

CudaGraph::CudaGraph(const core::Graph& G) {
    neighbours = std::vector<uint32_t>(G.edge_count());
    neighboursOffset = std::vector<uint32_t>(G.size() + 1);

    uint32_t offset = 0;
    for (uint32_t v = 0; v < G.size(); v++) {
        neighboursOffset[v] = offset;
        for (uint32_t u : G.get_neighbours(v)) {
            neighbours[offset++] = u;
        }
        std::sort(neighbours.begin() + neighboursOffset[v], neighbours.begin() + offset);
    }
    neighboursOffset.back() = offset;
    this->allocGPU();
}

void CudaGraph::allocGPU() {
    dev_neighboursOffset = cuda::malloc<uint32_t>(this->neighboursOffset.size());
    dev_neighbours = cuda::malloc<uint32_t>(this->neighbours.size());
    dev_size = cuda::malloc<uint32_t>(1);
    dev_edgeCount = cuda::malloc<uint32_t>(1);
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
    return this->dev_neighboursOffset[v + 1] - this->dev_neighboursOffset[v];
}

__host__ uint32_t CudaGraph::neighboursOut(uint32_t v) const {
    return this->neighboursOffset[v + 1] - this->neighboursOffset[v];
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::match(const core::Graph& bigGraph,
                                                              const core::Graph& smallGraph) {
    if (smallGraph.size() > bigGraph.size()) return std::nullopt;

    auto bigCudaGraph = CudaGraph(bigGraph);
    auto smallCudaGraph = CudaGraph(smallGraph);

    std::cout << "CREATED GRAPHJS\n";

    candidatesSizes_ = createCandidateLists(bigGraph, smallGraph, dev_candidates_);
    resultTable_ = ResultTable(smallGraph.size());
    std::cout << "CREATED CANDIDAETS\n";

    // Process first vertex
    uint32_t firstVertex = this->getNextVertex(smallGraph, candidatesSizes_);
    resultTable_.dev_data = dev_candidates_[firstVertex];
    resultTable_.map(firstVertex);
    resultTable_.rowCount = 1;
    std::cout << "PROCESSED FIRST VERTEX CANDIDAETS\n";

    for (int v = 1; v < smallGraph.size(); v++) {
        auto nextVertex = this->getNextVertex(smallGraph, candidatesSizes_);
        addVertexToResultTable(v, dev_candidates_[nextVertex], bigGraph, smallGraph);
        resultTable_.map(nextVertex);
    }

    std::cout << "PROCESSED ALL VERTEX CANDIDAETS\n";

    // return first row of result tablea
    auto result = obtainResult(resultTable_);

    // free everything remainingin;
    free(dev_candidates_);

    return result;
}

std::vector<uint32_t> CudaSubgraphMatcher::createCandidateLists(const CudaGraph& bigGraph, const CudaGraph& smallGraph,
                                                                uint32_t** dev_candidatesList) {
    auto candidateListsSizes = std::vector<uint32_t>(smallGraph.size());

    uint32_t num_blocks = (bigGraph.size() + block_size_ - 1) / block_size_;

    uint32_t* dev_candidateSet = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_prefixScan = cuda::malloc<uint32_t>(bigGraph.size());

    uint32_t* dev_candidateCount = cuda::malloc<uint32_t>(1);

    for (uint32_t u = 0; u < smallGraph.size(); u++) {
        std::cout << "processeed " << u << " candidate\n";
        cuda::memset<uint32_t>(dev_candidateSet, 0, bigGraph.size());
        createCandidateSetKernel<<<num_blocks, block_size_>>>(
            u, dev_candidateSet, bigGraph.size(), smallGraph.neighboursOut(u), bigGraph.dev_neighbours,
            bigGraph.dev_neighboursOffset, smallGraph.dev_neighbours, smallGraph.dev_neighboursOffset);
        cuda::ExclusiveSum<uint32_t>(dev_prefixScan, dev_candidateSet, bigGraph.size());
        cuda::memcpy_dev_host<uint32_t>(&candidateListsSizes[u], &dev_prefixScan[bigGraph.size() - 1], 1);

        uint32_t* dev_candidates = cuda::malloc<uint32_t>(candidateListsSizes[u]);
        filterCandidates<<<num_blocks, block_size_>>>(dev_candidates, dev_prefixScan, dev_candidateSet,
                                                      &candidateListsSizes[u]);
        cuda::radixSort<uint32_t>(dev_candidates, dev_candidates,
                                  candidateListsSizes[u]); // for speed up of set operations
        dev_candidatesList[u] = dev_candidates;
    }

    cuda::free(dev_candidateSet);
    cuda::free(dev_prefixScan);
    cuda::free(dev_candidateCount);
    return candidateListsSizes;
}

uint32_t CudaSubgraphMatcher::getNextVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidateListsSizes) {
    uint32_t highestScoreVertex = 0;
    uint32_t currentMax = 0;
    for (uint32_t v = 0; v < graph.size(); v++) {
        if (resultTable_.mapping[v] != UINT32_MAX) continue;
        if (candidateListsSizes[v] / graph.neighboursOut(v) > currentMax) {
            currentMax = candidateListsSizes[v] / graph.neighboursOut(v);
            highestScoreVertex = v;
        }
    }
    return highestScoreVertex;
}

std::vector<uint32_t> CudaSubgraphMatcher::getMappedNeighboursIn(int v, const CudaGraph& graph) {
    auto neighboursIn = std::vector<uint32_t>();

    for (uint32_t u = 0; u < graph.size(); u++) {
        if (resultTable_.mapping[u] == UINT32_MAX) continue;

        for (int i = 0; i < graph.neighboursOut(u); i++)
            if (graph.dev_neighbours[i + graph.dev_neighboursOffset[u]] == v) neighboursIn.push_back(u);
    }
    return neighboursIn;
}

void CudaSubgraphMatcher::addVertexToResultTable(int v, uint32_t* dev_candidates, const CudaGraph& bigGraph,
                                                 const CudaGraph& smallGraph) {
    auto neighboursIn = getMappedNeighboursIn(v, smallGraph);
    uint32_t* dev_GBA;
    auto GBAOffsets = allocateMemoryForJoining(v, dev_GBA, resultTable_, bigGraph);

    uint32_t* dev_GBAOffsets = cuda::malloc<uint32_t>(GBAOffsets.size());
    cuda::memcpy_host_dev<uint32_t>(dev_GBAOffsets, GBAOffsets.data(), GBAOffsets.size());

    uint32_t firstNeighbour = neighboursIn.front();
    uint32_t num_blocks = (GBAOffsets.size() - 1 + joiningBlockSize_ - 1) / joiningBlockSize_;

    uint32_t* dev_baseNeighbours =
        bigGraph.dev_neighbours + bigGraph.dev_neighboursOffset[resultTable_.mapping[firstNeighbour]];

    for (auto u : neighboursIn) {
        uint32_t bigUVertex = resultTable_.mapping[u];
        if (u == firstNeighbour) {
            joinResultTableRowFirst<<<num_blocks, joiningBlockSize_>>>(
                GBAOffsets.size() - 1, resultTable_, dev_GBA, dev_GBAOffsets,
                bigGraph.dev_neighbours + bigGraph.dev_neighboursOffset[bigUVertex], dev_candidates,
                candidatesSizes_[v]);
        } else {
            uint32_t* dev_currentNeighbours =
                bigGraph.dev_neighbours + bigGraph.dev_neighboursOffset[resultTable_.mapping[u]];
            uint32_t currentNeighboursSize = bigGraph.neighboursOut(bigUVertex);
            joinResultTableRowSecond<<<num_blocks, joiningBlockSize_>>>(GBAOffsets.size() - 1, dev_GBA, dev_GBAOffsets,
                                                                        dev_currentNeighbours, currentNeighboursSize,
                                                                        dev_baseNeighbours);
        }
    }
    linkGBAWithResult(dev_GBA, GBAOffsets, dev_GBAOffsets, resultTable_, dev_baseNeighbours);
}

void CudaSubgraphMatcher::linkGBAWithResult(uint32_t* dev_GBA, const std::vector<uint32_t>& GBAOffsets,
                                            uint32_t* dev_GBAOffsets, ResultTable& resultTable, uint32_t* neighbours) {
    uint32_t* dev_GBAPrefixScan = cuda::malloc<uint32_t>(GBAOffsets.back());
    cuda::ExclusiveSum<uint32_t>(dev_GBAPrefixScan, dev_GBA, GBAOffsets.back());
    uint32_t maxValue = 0;
    cuda::memcpy_dev_host<uint32_t>(&maxValue, &dev_GBAPrefixScan[GBAOffsets.back() - 1], 1);
    uint32_t* dev_newResultTableData = cuda::malloc<uint32_t>(maxValue);

    uint32_t num_blocks = (GBAOffsets.size() - 1 + joiningBlockSize_ - 1) / joiningBlockSize_;
    linkingKernel<<<num_blocks, joiningBlockSize_>>>(dev_newResultTableData, resultTable, dev_GBAPrefixScan, dev_GBA,
                                                     dev_GBAOffsets, neighbours);

    cuda::free(resultTable.dev_data);
    cuda::free(dev_GBA);
    cuda::free(dev_GBAOffsets);
    resultTable.dev_data = dev_newResultTableData;
}

std::vector<uint32_t> CudaSubgraphMatcher::allocateMemoryForJoining(int v, uint32_t*& GBA,
                                                                    const ResultTable& resultTable,
                                                                    const CudaGraph& bigGraph) {
    auto GBAOffsets = std::vector<uint32_t>(resultTable.rowCount + 1);
    auto mappedV = resultTable.mapping[v];

    GBAOffsets[0] = 0;
    for (uint32_t i = 0; i < resultTable.rowCount; i++) {
        GBAOffsets[i + 1] = GBAOffsets[i] + bigGraph.neighboursOut(mappedV + i * resultTable.size);
    }
    GBA = cuda::malloc<uint32_t>(GBAOffsets.back());
    cuda::memset<uint32_t>(GBA, 2, GBAOffsets.back()); // set everything to 2 to speed ub set operations
    return GBAOffsets;
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::obtainResult(const ResultTable& resultTable) {
    if (resultTable.rowCount == 0) return std::nullopt;

    auto result = std::vector<uint32_t>(resultTable.size);
    cuda::memcpy_dev_host<uint32_t>(result.data(), resultTable.dev_data, result.size());
    cuda::free(resultTable.dev_data);

    auto vertexResult = std::vector<vertex>(result.size());
    for (vertex v = 0; v < result.size(); v++) {
        vertexResult[v] = result[v];
    }
    return vertexResult;
}

} // namespace pattern