#include "graphManager.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <numeric>

GraphManager::GraphManager() : graph(0) {
    Initialize(std::move(graph));
}

void GraphManager::Initialize(core::Graph&& newGraph) {
    std::vector<std::pair<float, float>> vertexPositions;
    vertexPositions.reserve(newGraph.size());

    const float ANGLE_DIFF = 2.0f * M_PI / newGraph.size();
    for (vertex v = 0; v < newGraph.size(); v++) {
        vertexPositions.emplace_back(std::make_pair(cosf(v * ANGLE_DIFF), sinf(v * ANGLE_DIFF)));
    }

    Initialize(std::move(newGraph), std::move(vertexPositions));
}

void GraphManager::Initialize(core::Graph&& newGraph, std::vector<std::pair<float, float>>&& vertexPositions) {
    graph = newGraph;
    vertexPositions2D[0] = std::vector<float>(2 * graph.size());
    vertexPositions2D[1] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[0] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[1] = std::vector<float>(2 * graph.size());
    vertexStates = std::vector<unsigned int>(graph.size());

    float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest();
    for (vertex v = 0; v < graph.size(); v++) {
        auto [newX, newY] = vertexPositions[v];
        vertexPositions2D[readBufferId][2 * v] = newX;
        vertexPositions2D[readBufferId][2 * v + 1] = newY;
        vertexVelocities2D[readBufferId][2 * v] = vertexVelocities2D[readBufferId][2 * v + 1] = 0.0f;
        vertexStates[v] = 0;

        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    ResizeAnimationData();
    animationTimeLeftSeconds = std::nullopt;

    boundingWidth = renderedVertexCount <= 1 ? 1.0 : maxX - minX;
    boundingHeight = renderedVertexCount <= 1 ? 1.0 : maxY - minY;
    centerX = renderedVertexCount == 0 ? 0.0f : (minX + maxX) / 2;
    centerY = renderedVertexCount == 0 ? 0.0f : (minY + maxY) / 2;
}

void GraphManager::AlignNodes(std::vector<std::optional<std::pair<float, float>>>& positions2D) {
    for (vertex v = 0; v < graph.size(); v++) {
        if (positions2D[v].has_value()) {
            auto [x, y] = positions2D[v].value();
            vertexPositions2D[readBufferId][2 * v] = x;
            vertexPositions2D[readBufferId][2 * v + 1] = y;
            vertexStates[v] |= 0b10;
        }
    }

    animationTimeLeftSeconds = settings.alignmentAnimationTotalTimeSeconds;
}

void GraphManager::ResizeAnimationData() {
    vertexRenderedPositions2D.resize(2 * graph.size());
    tracking.resize(graph.size());
    std::iota(tracking.begin(), tracking.end(), 0);
    renderedVertexCount = graph.size();
}

void GraphManager::UpdateBounds(float deltaTimeSeconds) {
    float minX = std::numeric_limits<float>::max(), minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest(), maxY = std::numeric_limits<float>::lowest();
    for (vertex v = 0; v < renderedVertexCount; v++) {
        float newX = vertexRenderedPositions2D[2 * v];
        float newY = vertexRenderedPositions2D[2 * v + 1];

        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    auto goalBoundingWidth = renderedVertexCount <= 1 ? 1.0 : maxX - minX;
    auto goalBoundingHeight = renderedVertexCount <= 1 ? 1.0 : maxY - minY;
    auto goalCenterX = renderedVertexCount == 0 ? 0.0f : (minX + maxX) / 2;
    auto goalCenterY = renderedVertexCount == 0 ? 0.0f : (minY + maxY) / 2;

    auto delta = BOUNDS_MOVING_SPEED * deltaTimeSeconds;
    boundingWidth = Approach(boundingWidth, goalBoundingWidth, delta);
    boundingHeight = Approach(boundingHeight, goalBoundingHeight, delta);
    centerX = Approach(centerX, goalCenterX, delta);
    centerY = Approach(centerY, goalCenterY, delta);
}

float GraphManager::Approach(float value, float goal, float change) {
    if (change >= abs(value - goal)) return goal;
    return value + change * ((goal > value) - (value > goal));
}

void GraphManager::UpdatePositions(float deltaTimeSeconds, bool dragging) {
    for (unsigned int i = 0; i < graph.size(); i++) {
        float x = vertexPositions2D[readBufferId][2 * i];
        float y = vertexPositions2D[readBufferId][2 * i + 1];

        float acc_x = 0.0f;
        float acc_y = 0.0f;

        for (unsigned int j = 0; j < graph.size(); j++) {
            if (i == j) continue;

            float dx = x - vertexPositions2D[readBufferId][2 * j];
            float dy = y - vertexPositions2D[readBufferId][2 * j + 1];
            float dist = sqrtf(dx * dx + dy * dy);
            if (dist < MIN_NODE_DISTANCE) dist = MIN_NODE_DISTANCE;

            if (graph.has_edge(i, j) || graph.has_edge(j, i)) {
                float springCoefficient = -settings.springStrength * logf(dist / settings.springLength) / dist;
                acc_x += springCoefficient * dx;
                acc_y += springCoefficient * dy;
            }

            float repulsionCoefficient = settings.nodeRepulsion / (dist * dist * dist);
            acc_x += repulsionCoefficient * dx;
            acc_y += repulsionCoefficient * dy;
        }

        float vel_x = vertexVelocities2D[readBufferId][2 * i];
        float vel_y = vertexVelocities2D[readBufferId][2 * i + 1];

        if (vertexStates[i] & 0b10u || dragging && vertexStates[i] & 0b01u) vel_x = vel_y = 0.0f;

        acc_x -= settings.nodeDrag * vel_x;
        acc_y -= settings.nodeDrag * vel_y;

        vertexVelocities2D[1 - readBufferId][2 * i] = vel_x + acc_x * deltaTimeSeconds;
        vertexVelocities2D[1 - readBufferId][2 * i + 1] = vel_y + acc_y * deltaTimeSeconds;
        vertexPositions2D[1 - readBufferId][2 * i] = x + vel_x * deltaTimeSeconds;
        vertexPositions2D[1 - readBufferId][2 * i + 1] = y + vel_y * deltaTimeSeconds;
    }

    readBufferId = 1 - readBufferId;
}

bool GraphManager::UpdateRenderedPositions(float deltaTimeSeconds) {
    auto animationFinished = false;
    if (animationTimeLeftSeconds.has_value() && animationTimeLeftSeconds.value() < deltaTimeSeconds) {
        ResizeAnimationData();
        animationTimeLeftSeconds = std::nullopt;
        animationFinished = true;
    }

    if (IsAnimationRunning()) {
        for (vertex v = 0; v < renderedVertexCount; v++) {
            auto dx = vertexPositions2D[readBufferId][2 * tracking[v]] - vertexRenderedPositions2D[2 * v];
            auto dy = vertexPositions2D[readBufferId][2 * tracking[v] + 1] - vertexRenderedPositions2D[2 * v + 1];
            vertexRenderedPositions2D[2 * v] += deltaTimeSeconds * dx / animationTimeLeftSeconds.value();
            vertexRenderedPositions2D[2 * v + 1] += deltaTimeSeconds * dy / animationTimeLeftSeconds.value();
        }
        animationTimeLeftSeconds.value() -= deltaTimeSeconds;
    } else {
        for (vertex v = 0; v < renderedVertexCount; v++) {
            vertexRenderedPositions2D[2 * v] = vertexPositions2D[readBufferId][2 * tracking[v]];
            vertexRenderedPositions2D[2 * v + 1] = vertexPositions2D[readBufferId][2 * tracking[v] + 1];
        }
    }

    return animationFinished;
}

bool GraphManager::IsAnimationRunning() const {
    return animationTimeLeftSeconds.has_value() && animationTimeLeftSeconds.value() > 0.0f;
}

vertex GraphManager::RenderedVertexCount() const {
    return renderedVertexCount;
}

bool GraphManager::AddVertex(float x, float y) {
    auto v = graph.add_vertex();
    vertexPositions2D[readBufferId].push_back(x);
    vertexPositions2D[readBufferId].push_back(y);
    vertexPositions2D[1 - readBufferId].push_back(0.0f);
    vertexPositions2D[1 - readBufferId].push_back(0.0f);
    vertexVelocities2D[readBufferId].push_back(0.0f);
    vertexVelocities2D[readBufferId].push_back(0.0f);
    vertexVelocities2D[1 - readBufferId].push_back(0.0f);
    vertexVelocities2D[1 - readBufferId].push_back(0.0f);
    vertexStates.push_back(0b01u);

    ResizeAnimationData();
    return true;
}

void GraphManager::ClearSelection() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        vertexStates[i] &= 0b10u;
    }
}

std::vector<vertex> GraphManager::GetCollidingNodes(float x, float y, float nodeRadius) const {
    std::vector<vertex> collidingNodes;
    for (vertex v = 0; v < renderedVertexCount; v++) {
        float dx = x - vertexRenderedPositions2D[2 * v];
        float dy = y - vertexRenderedPositions2D[2 * v + 1];
        if (dx * dx + dy * dy <= nodeRadius * nodeRadius) collidingNodes.push_back(v);
    }
    return collidingNodes;
}

std::vector<vertex> GraphManager::GetCollidingNodes(float startX, float startY, float endX, float endY) const {
    if (startX > endX) std::swap(startX, endX);
    if (startY > endY) std::swap(startY, endY);

    std::vector<vertex> collidingNodes;
    for (vertex v = 0; v < renderedVertexCount; v++) {
        float vertexX = vertexRenderedPositions2D[2 * v];
        float vertexY = vertexRenderedPositions2D[2 * v + 1];

        if (vertexX < startX || vertexX > endX || vertexY < startY || vertexY > endY) continue;
        collidingNodes.push_back(v);
    }
    return collidingNodes;
}

void GraphManager::SelectNodes(const std::vector<vertex>& nodes) {
    for (auto v : nodes)
        vertexStates[tracking[v]] |= 0b01u;
}

void GraphManager::ToggleNodes(const std::vector<vertex>& nodes) {
    for (auto v : nodes)
        vertexStates[tracking[v]] ^= 0b01u;
}

void GraphManager::DeleteSelection() {
    std::vector<vertex> selectedVertices;

    for (unsigned int i = graph.size(); i > 0; i--) {
        vertex v = i - 1;
        if (vertexStates[v] & 0b01u) {
            selectedVertices.push_back(v);
            vertexPositions2D[0].erase(vertexPositions2D[0].begin() + 2 * v, vertexPositions2D[0].begin() + 2 * v + 2);
            vertexPositions2D[1].erase(vertexPositions2D[1].begin() + 2 * v, vertexPositions2D[1].begin() + 2 * v + 2);
            vertexVelocities2D[0].erase(vertexVelocities2D[0].begin() + 2 * v,
                                        vertexVelocities2D[0].begin() + 2 * v + 2);
            vertexVelocities2D[1].erase(vertexVelocities2D[1].begin() + 2 * v,
                                        vertexVelocities2D[1].begin() + 2 * v + 2);
            vertexStates.erase(vertexStates.begin() + v);
        }
    }

    graph.remove_vertices(selectedVertices);

    ResizeAnimationData();
}

void GraphManager::ConnectSelection() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (vertexStates[i] & vertexStates[j] & 0b01u) {
                graph.add_edge(i, j);
                graph.add_edge(j, i);
            }
        }
    }
}

void GraphManager::ConnectNodes(vertex start, vertex end) {
    if (tracking[start] == tracking[end]) return;
    graph.add_edge(tracking[start], tracking[end]);
    graph.add_edge(tracking[end], tracking[start]);
}

void GraphManager::DisconnectSelection() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (vertexStates[i] & vertexStates[j] & 0b01u) {
                graph.remove_edge(i, j);
                graph.remove_edge(j, i);
            }
        }
    }
}

void GraphManager::ContractSelection() {

    auto size = graph.size();
    std::vector<vertex> selectedVertices;
    bool* visited = new bool[size];

    // 1. Preprocessing
    for (unsigned int i = size; i > 0; i--) {
        vertex v = i - 1;
        if (vertexStates[v] & 0b01u) {
            selectedVertices.emplace_back(v);
        }
        visited[v] = !(vertexStates[v] & 0b01u);
    }

    // 2. Add vertex for each connected component and edges between them.
    std::vector<vertex> toVisit;
    std::vector<vertex> nonContractedEnds;
    std::vector<vertex> contractedVertices;
    for (vertex v = 0; v < size; v++) {
        if (visited[v]) continue;
        toVisit.emplace_back(v);
        float x = 0.0f, y = 0.0f, v_x = 0.0f, v_y = 0.0f;
        unsigned int ccSize = 0;
        while (!toVisit.empty()) {
            vertex u = toVisit.back();
            toVisit.pop_back();
            visited[u] = true;
            x += vertexPositions2D[readBufferId][2 * u];
            y += vertexPositions2D[readBufferId][2 * u + 1];
            v_x += vertexVelocities2D[readBufferId][2 * u];
            v_y += vertexVelocities2D[readBufferId][2 * u + 1];
            ccSize++;
            contractedVertices.emplace_back(u);
            for (auto w : graph.neighbours(u)) {
                if (!(vertexStates[w] & 0b01u)) {
                    nonContractedEnds.emplace_back(w);
                }
                if (!visited[w]) {
                    toVisit.emplace_back(w);
                }
            }
        }

        auto uniqueEnd = std::unique(nonContractedEnds.begin(), nonContractedEnds.end());
        nonContractedEnds.erase(uniqueEnd, nonContractedEnds.end());

        vertex cc = graph.add_vertex();
        vertexPositions2D[readBufferId].push_back(x / ccSize);
        vertexPositions2D[readBufferId].push_back(y / ccSize);
        vertexPositions2D[1 - readBufferId].push_back(0.0f);
        vertexPositions2D[1 - readBufferId].push_back(0.0f);
        vertexVelocities2D[readBufferId].push_back(v_x / ccSize);
        vertexVelocities2D[readBufferId].push_back(v_y / ccSize);
        vertexVelocities2D[1 - readBufferId].push_back(0.0f);
        vertexVelocities2D[1 - readBufferId].push_back(0.0f);
        vertexStates.push_back(0b01u);

        while (!contractedVertices.empty()) {
            vertex u = contractedVertices.back();
            contractedVertices.pop_back();
            tracking[u] = cc;
        }

        while (!nonContractedEnds.empty()) {
            vertex u = nonContractedEnds.back();
            nonContractedEnds.pop_back();
            graph.add_edge(cc, u);
            graph.add_edge(u, cc);
        }
    }

    // 3. Remove vertices from each connected component
    int* toBeRemoved = new int[graph.size()]{false};
    for (auto v : selectedVertices) {
        toBeRemoved[v] = true;
        vertexPositions2D[0].erase(vertexPositions2D[0].begin() + 2 * v, vertexPositions2D[0].begin() + 2 * v + 2);
        vertexPositions2D[1].erase(vertexPositions2D[1].begin() + 2 * v, vertexPositions2D[1].begin() + 2 * v + 2);
        vertexVelocities2D[0].erase(vertexVelocities2D[0].begin() + 2 * v, vertexVelocities2D[0].begin() + 2 * v + 2);
        vertexVelocities2D[1].erase(vertexVelocities2D[1].begin() + 2 * v, vertexVelocities2D[1].begin() + 2 * v + 2);
        vertexStates.erase(vertexStates.begin() + v);
    }
    vertex* vertexIndexDelta = new vertex[graph.size() + 1];
    vertexIndexDelta[0] = 0;
    std::partial_sum(toBeRemoved, toBeRemoved + graph.size(), vertexIndexDelta + 1);
    std::transform(tracking.begin(), tracking.end(), tracking.begin(),
                   [vertexIndexDelta](vertex v) { return v - vertexIndexDelta[v]; });
    graph.remove_vertices(selectedVertices, toBeRemoved, vertexIndexDelta);

    animationTimeLeftSeconds = settings.contractionAnimationTotalTimeSeconds;

    delete[] visited;
    delete[] toBeRemoved;
    delete[] vertexIndexDelta;
}

void GraphManager::SubdivideSelection() {
    vertex originalSize = graph.size();
    for (unsigned int i = 0; i < originalSize; i++) {
        for (unsigned int j = 0; j < i; j++) {
            if ((vertexStates[i] & vertexStates[j] & 0b01u) && graph.has_edge(i, j) && graph.has_edge(j, i)) {

                graph.remove_edge(i, j);
                graph.remove_edge(j, i);
                vertex k = graph.add_vertex();
                graph.add_edge(i, k);
                graph.add_edge(k, i);
                graph.add_edge(j, k);
                graph.add_edge(k, j);

                float x = 0.5 * (vertexPositions2D[readBufferId][2 * i] + vertexPositions2D[readBufferId][2 * j]);
                float y =
                    0.5 * (vertexPositions2D[readBufferId][2 * i + 1] + vertexPositions2D[readBufferId][2 * j + 1]);
                vertexPositions2D[readBufferId].push_back(x);
                vertexPositions2D[readBufferId].push_back(y);
                vertexPositions2D[1 - readBufferId].push_back(0.0f);
                vertexPositions2D[1 - readBufferId].push_back(0.0f);
                vertexVelocities2D[readBufferId].push_back(0.0f);
                vertexVelocities2D[readBufferId].push_back(0.0f);
                vertexVelocities2D[1 - readBufferId].push_back(0.0f);
                vertexVelocities2D[1 - readBufferId].push_back(0.0f);
                vertexStates.push_back(0);
            }
        }
    }

    ResizeAnimationData();
}

void GraphManager::AnchorSelection() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        if (vertexStates[i] & 0b01u) vertexStates[i] |= 0b10u;
    }
}

void GraphManager::FreeSelection() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        if (vertexStates[i] & 0b01u) vertexStates[i] &= 0b01u;
    }
}

const std::vector<float>& GraphManager::Positions2D() const {
    return vertexRenderedPositions2D;
}

const std::pair<float, float> GraphManager::Position2D(vertex v) const {
    return std::make_pair(vertexRenderedPositions2D[2 * v], vertexRenderedPositions2D[2 * v + 1]);
}

const std::vector<unsigned int> GraphManager::GetStates() const {
    std::vector<unsigned int> states(renderedVertexCount);
    for (unsigned int i = 0; i < renderedVertexCount; i++) {
        states[i] = vertexStates[tracking[i]];
    }
    return states;
}

const std::vector<unsigned int> GraphManager::GetEdges() const {
    std::vector<unsigned int> edges;
    for (unsigned int i = 0; i < renderedVertexCount; i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (graph.has_edge(tracking[i], tracking[j]) || graph.has_edge(tracking[j], tracking[i])) {
                edges.push_back(i);
                edges.push_back(j);
            }
        }
    }
    return edges;
}

const std::vector<unsigned int> GraphManager::GetRenderedLabelling(const std::vector<unsigned int>& labelling) const {
    std::vector<unsigned int> renderedLabelling(renderedVertexCount);

    for (unsigned int i = 0; i < renderedVertexCount; i++) {
        renderedLabelling[i] = labelling[tracking[i]];
    }

    return renderedLabelling;
}

const std::pair<float, float> GraphManager::BoundingSize() {
    return std::make_pair(boundingWidth, boundingHeight);
}

const std::pair<float, float> GraphManager::Center() {
    return std::make_pair(centerX, centerY);
}

const core::Graph& GraphManager::Graph() const {
    return graph;
}

void GraphManager::OnVertexDrag(float dx, float dy) {
    for (unsigned int i = 0; i < graph.size(); i++) {
        if (vertexStates[i] & 0b01u) {
            vertexPositions2D[readBufferId][2 * i] += dx;
            vertexPositions2D[readBufferId][2 * i + 1] += dy;
        }
    }
}

void GraphManager::Stop() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
    }
}

void GraphManager::UpdateSettings(GraphDrawingSettings settings) {
    this->settings = settings;
}
