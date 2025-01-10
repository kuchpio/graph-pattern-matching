#include "config.h"
#ifdef CUDA_ENABLED

#include "core.h"
#include "cuda_subgraph_matcher.h"
#include "cuda_helpers.cuh"

#include <optional>
#include <vector>
#include <algorithm>

#include <set>

#define NOT_MAPPED UINT32_MAX

namespace pattern
{

__device__ uint32_t binarySearch(const uint32_t* arr, int32_t size, int32_t target);

__global__ void getVmappingsKernel(uint32_t v_index, uint32_t* resultTableData, uint32_t resultTableSize,
                                   uint32_t resultTableRowCount, uint32_t* result) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > resultTableRowCount - 1) return;
    result[row] = resultTableData[row * resultTableSize + v_index];
}

__global__ void linkingKernel(uint32_t* dst, uint32_t* resultTableData, uint32_t resultTableSize,
                              uint32_t resultTableRowCount, uint32_t* GBAPreffixSum, uint32_t* GBA,
                              uint32_t* GBAOffsets, uint32_t* neighbours, uint32_t* neighboursOffset,
                              uint32_t firstVertex) {
    uint32_t row = blockIdx.x;
    uint32_t index = threadIdx.x;
    if (row >= resultTableRowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];
    auto vertexOffset = row * resultTableSize;
    auto baseNeighbours = neighbours + neighboursOffset[resultTableData[vertexOffset + firstVertex]];

    while (index < bufSize) {
        if (buf[index] == 1) {
            // copy
            for (uint32_t i = 0; i < resultTableSize; i++) {
                dst[(GBAPreffixSum[index + GBAOffsets[row]] - 1) * (resultTableSize + 1) + i] =
                    resultTableData[row * resultTableSize + i];
            }
            dst[(GBAPreffixSum[index + GBAOffsets[row]] - 1) * (resultTableSize + 1) + resultTableSize] =
                baseNeighbours[index];
        }
        index += blockDim.x;
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
    if (vNeighboursOut < uNeighboursOut) return;
    for (int i = 0; i < vNeighboursOut; i++) {
        auto neighbour = bigGraphNeighbours[i + bigGraphOffsets[v]];
        vNeighboursDegrees += bigGraphOffsets[neighbour + 1] - bigGraphOffsets[neighbour];
    }
    for (int i = 0; i < uNeighboursOut; i++) {
        auto neighbour = smallGraphNeighbours[i + smallGraphOffsets[u]];
        uNeighboursDegrees += smallGraphOffsets[neighbour + 1] - smallGraphOffsets[neighbour];
    }

    if (uNeighboursDegrees <= vNeighboursDegrees) candidateSet[v] = 1;
}

__global__ void filterCandidates(uint32_t* candidates, uint32_t* prefixScan, uint32_t* candidateSet, uint32_t size) {
    uint32_t v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= size) return;
    if (candidateSet[v]) candidates[prefixScan[v] - 1] = v;
}

__global__ void joinResultTableRowFirst(uint32_t v, uint32_t rowCount, uint32_t* resultTableData,
                                        uint32_t resultTableSize, uint32_t* GBA, uint32_t* GBAOffsets,
                                        uint32_t* neighbours, uint32_t* neighboursOffset, uint32_t* candidates,
                                        uint32_t candidatesSize) {
    uint32_t row = blockIdx.x;
    uint32_t index = threadIdx.x;

    if (row >= rowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];

    auto mOffset = row * resultTableSize;
    auto rowNeighbours = neighbours + neighboursOffset[resultTableData[v + row * resultTableSize]];

    // extern __shared__ uint32_t sharedRow[];

    // use shared memory here :)
    // do N(v) - m_i
    while (index < bufSize) {
        uint32_t mIndex = 0;
        buf[index] = 1;
        while (mIndex < resultTableSize) {
            if (resultTableData[mOffset + mIndex++] == rowNeighbours[index]) {
                buf[index] = 0;
                break;
            }
        }
        index += blockDim.x;
    }

    index = threadIdx.x;
    // buf(i) & C(v)
    while (index < bufSize) {

        if (buf[index] == 0) {
            index += blockDim.x;
            continue;
        }
        if (binarySearch(candidates, candidatesSize, rowNeighbours[index]) == UINT32_MAX) {
            buf[index] = 0;
        }
        index += blockDim.x;
    }
}

__global__ void joinResultTableRowSecond(uint32_t firstVertex, uint32_t currentVertex, uint32_t rowCount,
                                         uint32_t* resultTableData, uint32_t resultTableSize, uint32_t* GBA,
                                         uint32_t* GBAOffsets, uint32_t* neighbours, uint32_t* neighboursOffset) {
    uint32_t row = blockIdx.x;
    uint32_t index = threadIdx.x;

    if (row >= rowCount) return;

    uint32_t bufSize = GBAOffsets[row + 1] - GBAOffsets[row];
    uint32_t* buf = GBA + GBAOffsets[row];

    auto vertexOffset = row * resultTableSize;
    auto rowNeighbours = neighbours + neighboursOffset[resultTableData[vertexOffset + currentVertex]];
    auto baseNeighbours = neighbours + neighboursOffset[resultTableData[vertexOffset + firstVertex]];
    auto currentNeighboursSize = neighboursOffset[resultTableData[vertexOffset + currentVertex] + 1] -
                                 neighboursOffset[resultTableData[vertexOffset + currentVertex]];

    // extern __shared__ uint32_t sharedRow[];
    //  buf(i) & N(v)
    while (index < bufSize) {
        if (buf[index] == 0) {
            index += blockDim.x;
            continue;
        }
        if (binarySearch(rowNeighbours, currentNeighboursSize, baseNeighbours[index]) == UINT32_MAX) {
            buf[index] = 0;
        }
        index += blockDim.x;
    }
}

__device__ uint32_t binarySearch(const uint32_t* arr, int32_t size, int32_t target) {
    int32_t left = 0, right = size - 1;

    while (left <= right) {
        int32_t mid = left + (right - left) / 2;

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

__device__ uint32_t CudaGraph::dev_neighboursOut(uint32_t v) const {
    return this->dev_neighboursOffset[v + 1] - this->dev_neighboursOffset[v];
}

__host__ uint32_t CudaGraph::neighboursOut(uint32_t v) const {
    return this->neighboursOffset[v + 1] - this->neighboursOffset[v];
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::match(const core::Graph& bigGraph,
                                                              const core::Graph& smallGraph) {
    if (smallGraph.size() == 0 || bigGraph.size() == 0) return std::nullopt;
    if (smallGraph.size() > bigGraph.size()) return std::nullopt;

    auto bigCudaGraph = CudaGraph(bigGraph);
    auto smallCudaGraph = CudaGraph(smallGraph);

    // Create Candidates
    auto candidates = std::vector<uint32_t*>();
    auto candidatesSizesStatus = createCandidateLists(bigGraph, smallGraph, candidates);
    if (!candidatesSizesStatus) {
        return std::nullopt;
    }
    auto candidatesSizes = candidatesSizesStatus.value();

    auto resultTable = ResultTable(smallGraph.size());

    // Process first vertex
    uint32_t firstVertex = this->getFirstVertex(smallGraph, candidatesSizes);
    resultTable.dev_data = candidates[firstVertex];
    resultTable.map(firstVertex);
    resultTable.rowCount = candidatesSizes[firstVertex];

    for (int v = 1; v < smallGraph.size(); v++) {
        auto nextVertex = this->getNextVertex(smallGraph, candidatesSizes, resultTable.mapping).value();
        if (!addVertexToResultTable(nextVertex, candidates[nextVertex], candidatesSizes[nextVertex], bigGraph,
                                    smallGraph, resultTable)) {
            resultTable.map(nextVertex);
            freeNotMappedCandidates(resultTable.mapping, candidates);
            return std::nullopt;
        }
        resultTable.map(nextVertex);
    }

    return obtainResult(resultTable, bigCudaGraph);
}

std::optional<std::vector<uint32_t>> CudaSubgraphMatcher::createCandidateLists(const CudaGraph& bigGraph,
                                                                               const CudaGraph& smallGraph,
                                                                               std::vector<uint32_t*>& candidatesList) {
    auto success = true;
    auto candidateListsSizes = std::vector<uint32_t>(smallGraph.size());
    uint32_t num_blocks = (bigGraph.size() + block_size_ - 1) / block_size_;

    uint32_t* dev_candidateSet = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_prefixScan = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_tempCandidates = cuda::malloc<uint32_t>(bigGraph.size());
    uint32_t* dev_candidateCount = cuda::malloc<uint32_t>(1);

    for (uint32_t u = 0; u < smallGraph.size(); u++) {
        cuda::memset<uint32_t>(dev_candidateSet, 0, bigGraph.size());
        cuda::memset<uint32_t>(dev_tempCandidates, 0, bigGraph.size());
        createCandidateSetKernel<<<num_blocks, block_size_>>>(
            u, dev_candidateSet, bigGraph.size(), smallGraph.neighboursOut(u), bigGraph.dev_neighbours,
            bigGraph.dev_neighboursOffset, smallGraph.dev_neighbours, smallGraph.dev_neighboursOffset);
        cuda::InclusiveSum<uint32_t>(dev_prefixScan, dev_candidateSet, bigGraph.size());
        cuda::memcpy_dev_host<uint32_t>(&candidateListsSizes[u], &dev_prefixScan[bigGraph.size() - 1], 1);

        // If u has no candidates free previously allocated candidates and return false
        if (candidateListsSizes[u] == 0) {
            for (auto candidate : candidatesList)
                cuda::free(candidate);
            success = false;
            break;
        }

        filterCandidates<<<num_blocks, block_size_>>>(dev_tempCandidates, dev_prefixScan, dev_candidateSet,
                                                      bigGraph.size());
        uint32_t* dev_candidates = cuda::malloc<uint32_t>(candidateListsSizes[u]);
        cuda::radixSort<uint32_t>(dev_candidates, dev_tempCandidates,
                                  candidateListsSizes[u]); // for speed up of set operations
        candidatesList.push_back(dev_candidates);
    }

    // Free memory
    cuda::free(dev_candidateSet);
    cuda::free(dev_prefixScan);
    cuda::free(dev_candidateCount);
    cuda::free(dev_tempCandidates);
    if (success == false) return std::nullopt;

    return candidateListsSizes;
}

uint32_t CudaSubgraphMatcher::getFirstVertex(const CudaGraph& graph, const std::vector<uint32_t>& candidateListsSizes) {
    uint32_t highestScoreVertex = 0;
    double currentMin = DBL_MAX;
    for (uint32_t v = 0; v < graph.size(); v++) {
        if ((static_cast<double>(candidateListsSizes[v]) / static_cast<double>(graph.neighboursOut(v))) < currentMin) {
            currentMin = static_cast<double>(candidateListsSizes[v]) / static_cast<double>(graph.neighboursOut(v));
            highestScoreVertex = v;
        }
    }
    return highestScoreVertex;
}

std::optional<uint32_t> CudaSubgraphMatcher::getNextVertex(const CudaGraph& graph,
                                                           const std::vector<uint32_t>& candidateListsSizes,
                                                           const std::vector<uint32_t>& mapping) {
    uint32_t highestScoreVertex = UINT32_MAX;
    double currentMax = DBL_MAX;
    for (uint32_t v = 0; v < graph.size(); v++) {
        if (mapping[v] != NOT_MAPPED) continue;
        if ((static_cast<double>(candidateListsSizes[v]) / static_cast<double>(graph.neighboursOut(v))) < currentMax) {
            bool connected = false;
            for (int i = 0; i < mapping.size(); i++) {
                if (mapping[i] != NOT_MAPPED) {
                    if (graph.hasEdge(i, v)) connected = true;
                }
            }
            if (connected == false) continue;
            currentMax = static_cast<double>(candidateListsSizes[v]) / static_cast<double>(graph.neighboursOut(v));
            highestScoreVertex = v;
        }
    }
    if (highestScoreVertex == UINT32_MAX) return std::nullopt;
    return highestScoreVertex;
}

std::vector<uint32_t> CudaSubgraphMatcher::getMappedNeighboursIn(int v, const CudaGraph& graph,
                                                                 const std::vector<uint32_t>& mapping) {
    auto neighboursIn = std::vector<uint32_t>();

    for (uint32_t u = 0; u < graph.size(); u++) {
        if (mapping[u] == NOT_MAPPED) continue;

        for (int i = 0; i < graph.neighboursOut(u); i++)
            if (graph.neighbours[i + graph.neighboursOffset[u]] == v) neighboursIn.push_back(u);
    }
    return neighboursIn;
}

std::optional<ResultTable> CudaSubgraphMatcher::addVertexToResultTable(uint32_t v, uint32_t* dev_candidates,
                                                                       uint32_t vCandidatesCount,
                                                                       const CudaGraph& bigGraph,
                                                                       const CudaGraph& smallGraph,
                                                                       ResultTable& resultTable) {
    auto neighboursIn = getMappedNeighboursIn(v, smallGraph, resultTable.mapping);
    auto minNeighourIn =
        std::min_element(neighboursIn.begin(), neighboursIn.end(), [&smallGraph](uint32_t left, uint32_t right) {
            return smallGraph.neighboursOut(left) < smallGraph.neighboursOut(right);
        });
    uint32_t* dev_GBA;
    auto GBAOffsets = allocateMemoryForJoining(*minNeighourIn, dev_GBA, resultTable, bigGraph);

    uint32_t* dev_GBAOffsets = cuda::malloc<uint32_t>(GBAOffsets.size());
    cuda::memcpy_host_dev<uint32_t>(dev_GBAOffsets, GBAOffsets.data(), GBAOffsets.size());
    uint32_t firstNeighbour = neighboursIn.front();

    for (auto u : neighboursIn) {
        uint32_t uMapping = resultTable.mapping[u];
        if (u == firstNeighbour) {
            joinResultTableRowFirst<<<resultTable.rowCount, joiningBlockSize_>>>(
                uMapping, resultTable.rowCount, resultTable.dev_data, resultTable.size, dev_GBA, dev_GBAOffsets,
                bigGraph.dev_neighbours, bigGraph.dev_neighboursOffset, dev_candidates, vCandidatesCount);
        } else {
            joinResultTableRowSecond<<<resultTable.rowCount, joiningBlockSize_>>>(
                resultTable.mapping[firstNeighbour], uMapping, resultTable.rowCount, resultTable.dev_data,
                resultTable.size, dev_GBA, dev_GBAOffsets, bigGraph.dev_neighbours, bigGraph.dev_neighboursOffset);
        }
    }
    cuda::free(dev_candidates);
    return linkGBAWithResult(dev_GBA, GBAOffsets, dev_GBAOffsets, resultTable, bigGraph,
                             resultTable.mapping[firstNeighbour]);
}

std::optional<ResultTable> CudaSubgraphMatcher::linkGBAWithResult(uint32_t* dev_GBA,
                                                                  const std::vector<uint32_t>& GBAOffsets,
                                                                  uint32_t* dev_GBAOffsets, ResultTable& resultTable,
                                                                  const CudaGraph& graph, uint32_t baseIndex) {

    // Get number of rows in the new table
    uint32_t* dev_GBAPrefixScan = cuda::malloc<uint32_t>(GBAOffsets.back());
    cuda::InclusiveSum<uint32_t>(dev_GBAPrefixScan, dev_GBA, GBAOffsets.back());
    uint32_t rowCount = 0;
    cuda::memcpy_dev_host<uint32_t>(&rowCount, &dev_GBAPrefixScan[GBAOffsets.back() - 1], 1);
    uint32_t* dev_newResultTableData;
    if (rowCount > 0) {
        dev_newResultTableData = cuda::malloc<uint32_t>(rowCount * (resultTable.size + 1));
        linkingKernel<<<GBAOffsets.size() - 1, joiningBlockSize_>>>(
            dev_newResultTableData, resultTable.dev_data, resultTable.size, resultTable.rowCount, dev_GBAPrefixScan,
            dev_GBA, dev_GBAOffsets, graph.dev_neighbours, graph.dev_neighboursOffset, baseIndex);
    }
    cuda::free(resultTable.dev_data);
    cuda::free(dev_GBA);
    cuda::free(dev_GBAOffsets);
    cuda::free(dev_GBAPrefixScan);
    if (rowCount == 0) return std::nullopt;

    resultTable.rowCount = rowCount;
    resultTable.dev_data = dev_newResultTableData;
    return resultTable;
}

std::vector<uint32_t> CudaSubgraphMatcher::allocateMemoryForJoining(int v, uint32_t*& GBA,
                                                                    const ResultTable& resultTable,
                                                                    const CudaGraph& bigGraph) {
    auto GBAOffsets = std::vector<uint32_t>(resultTable.rowCount + 1);
    auto mappedIndex = resultTable.mapping[v];
    std::vector<uint32_t> vMappings = std::vector<uint32_t>(resultTable.rowCount);
    uint32_t* dev_mappings = cuda::malloc<uint32_t>(vMappings.size());

    const uint32_t blockCount = (resultTable.rowCount + block_size_ - 1) / block_size_;
    getVmappingsKernel<<<blockCount, block_size_>>>(mappedIndex, resultTable.dev_data, resultTable.size,
                                                    resultTable.rowCount, dev_mappings);
    cuda::memcpy_dev_host<uint32_t>(vMappings.data(), dev_mappings, vMappings.size());
    cuda::free(dev_mappings);

    GBAOffsets[0] = 0;
    for (uint32_t i = 0; i < resultTable.rowCount; i++) {

        GBAOffsets[i + 1] = GBAOffsets[i] + bigGraph.neighboursOut(vMappings[i]);
    }
    GBA = cuda::malloc<uint32_t>(GBAOffsets.back());
    return GBAOffsets;
}

std::optional<std::vector<vertex>> CudaSubgraphMatcher::obtainResult(const ResultTable& resultTable,
                                                                     const CudaGraph& bigGraph) {
    if (resultTable.rowCount == 0) return std::nullopt;
    auto result = std::vector<uint32_t>(resultTable.size);
    cuda::memcpy_dev_host<uint32_t>(result.data(), resultTable.dev_data, result.size());
    cuda::free(resultTable.dev_data);

    auto vertexResult = std::vector<vertex>(bigGraph.size(), SIZE_MAX);
    for (vertex v = 0; v < result.size(); v++) {
        vertexResult[result[v]] = resultTable.vertexOrdering[v];
    }
    return vertexResult;
}

void ResultTable::print() {
    std::vector<uint32_t> data = std::vector<uint32_t>(rowCount * size);
    cuda::memcpy_dev_host<uint32_t>(data.data(), dev_data, data.size());
    printf("\nResultTable: ");
    for (int i = 0; i < rowCount; i++) {
        printf("\n row %u == [", i);
        for (int j = 0; j < size; j++) {
            printf("%u ", data[i * size + j]);
        }
        printf("]\n");
    }
}

void CudaSubgraphMatcher::freeNotMappedCandidates(const std::vector<uint32_t>& mapping,
                                                  const std::vector<uint32_t*>& candidates) {
    for (uint32_t mapIndex = 0; mapIndex < mapping.size(); ++mapIndex) {
        if (mapping[mapIndex] == NOT_MAPPED) {
            cuda::free(candidates[mapIndex]);
        }
    }
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
} // namespace pattern
#endif