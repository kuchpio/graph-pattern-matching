#include "graphManager.h"

#include <math.h>
#include <cfloat>
#include <algorithm>

GraphManager::GraphManager() : graph(0), boundingWidth(0), boundingHeight(0), centerX(0), centerY(0), dragging(false) {
}

void GraphManager::Initialize(core::Graph&& newGraph) {
    graph = newGraph;
    vertexPositions2D[0] = std::vector<float>(2 * graph.size());
    vertexPositions2D[1] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[0] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[1] = std::vector<float>(2 * graph.size());
    vertexStates = std::vector<unsigned int>(graph.size());

    float minX = FLT_MAX, minY = FLT_MAX, maxX = FLT_MIN, maxY = FLT_MIN;
    for (unsigned int i = 0; i < graph.size(); i++) {
        float newX = vertexPositions2D[readBufferId][2 * i] = 2 * ((float)rand() / RAND_MAX) - 1;
        float newY = vertexPositions2D[readBufferId][2 * i + 1] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
        vertexStates[i] = 0;

        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    boundingWidth = maxX - minX;
    boundingHeight = maxY - minY;
    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
    dragging = false;
}

void GraphManager::Initialize(core::Graph&& newGraph, std::vector<std::pair<float, float>>&& vertexPositions) {
    graph = newGraph;
    vertexPositions2D[0] = std::vector<float>(2 * graph.size());
    vertexPositions2D[1] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[0] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[1] = std::vector<float>(2 * graph.size());
    vertexStates = std::vector<unsigned int>(graph.size());

    float minX = FLT_MAX, minY = FLT_MAX, maxX = FLT_MIN, maxY = FLT_MIN;
    for (unsigned int i = 0; i < graph.size(); i++) {
        auto [newX, newY] = vertexPositions[i];
        vertexPositions2D[readBufferId][2 * i] = newX;
        vertexPositions2D[readBufferId][2 * i + 1] = newY;
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
        vertexStates[i] = 0;

        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    boundingWidth = maxX - minX;
    boundingHeight = maxY - minY;
    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
    dragging = false;
}

void GraphManager::UpdatePositions(float deltaTimeSeconds) {
    float minX = FLT_MAX, minY = FLT_MAX, maxX = FLT_MIN, maxY = FLT_MIN;

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

            if (graph.has_edge(i, j) || graph.has_edge(j, i)) {
                float springCoefficient = -SPRING_STRENGTH * logf(dist / SPRING_LENGTH) / dist;
                acc_x += springCoefficient * dx;
                acc_y += springCoefficient * dy;
            }

            float repulsionCoefficient = REPULSION_STRENGTH / (dist * dist * dist);
            acc_x += repulsionCoefficient * dx;
            acc_y += repulsionCoefficient * dy;
        }

        float vel_x = vertexVelocities2D[readBufferId][2 * i];
        float vel_y = vertexVelocities2D[readBufferId][2 * i + 1];

        if (vertexStates[i] & 0b10u || dragging && vertexStates[i] & 0b01u) vel_x = vel_y = 0.0f;

        acc_x -= DRAG * vel_x;
        acc_y -= DRAG * vel_y;

        vertexVelocities2D[1 - readBufferId][2 * i] = vel_x + acc_x * deltaTimeSeconds;
        vertexVelocities2D[1 - readBufferId][2 * i + 1] = vel_y + acc_y * deltaTimeSeconds;
        float newX = vertexPositions2D[1 - readBufferId][2 * i] = x + vel_x * deltaTimeSeconds;
        float newY = vertexPositions2D[1 - readBufferId][2 * i + 1] = y + vel_y * deltaTimeSeconds;

        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    readBufferId = 1 - readBufferId;

    if (dragging) return;
    boundingWidth = maxX - minX;
    boundingHeight = maxY - minY;
    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
}

bool GraphManager::HandleClick(float x, float y, float nodeRadius, bool isCtrl, bool newVertexRequested) {
    unsigned int selectedCtr = 0;
    for (unsigned int i = 0; i < graph.size(); i++) {
        float dx = x - vertexPositions2D[readBufferId][2 * i];
        float dy = y - vertexPositions2D[readBufferId][2 * i + 1];
        if (!isCtrl) vertexStates[i] &= 0b10u;
        if (dx * dx + dy * dy <= nodeRadius * nodeRadius) {
            vertexStates[i] ^= 0b01u;
            selectedCtr++;
        }
    }
    if (selectedCtr == 0 && newVertexRequested) {
        AddVertex(x, y);
        return true;
    }
    return false;
}

void GraphManager::AddVertex(float x, float y) {
    graph.add_vertex();
    vertexPositions2D[readBufferId].push_back(x);
    vertexPositions2D[readBufferId].push_back(y);
    vertexPositions2D[1 - readBufferId].push_back(0.0f);
    vertexPositions2D[1 - readBufferId].push_back(0.0f);
    vertexVelocities2D[readBufferId].push_back(0.0f);
    vertexVelocities2D[readBufferId].push_back(0.0f);
    vertexVelocities2D[1 - readBufferId].push_back(0.0f);
    vertexVelocities2D[1 - readBufferId].push_back(0.0f);
    vertexStates.push_back(0b01u);
}

void GraphManager::DeleteSelection() {
    std::vector<vertex> toBeRemoved;
    for (unsigned int i = graph.size(); i > 0; i--) {
        vertex v = i - 1;
        if (vertexStates[v] & 0b01u) {
            toBeRemoved.push_back(v);

            vertexPositions2D[0].erase(vertexPositions2D[0].begin() + 2 * v, vertexPositions2D[0].begin() + 2 * v + 2);
            vertexPositions2D[1].erase(vertexPositions2D[1].begin() + 2 * v, vertexPositions2D[1].begin() + 2 * v + 2);
            vertexVelocities2D[0].erase(vertexVelocities2D[0].begin() + 2 * v, vertexVelocities2D[0].begin() + 2 * v + 2);
            vertexVelocities2D[1].erase(vertexVelocities2D[1].begin() + 2 * v, vertexVelocities2D[1].begin() + 2 * v + 2);
            vertexStates.erase(vertexStates.begin() + v);
        }
    }
    graph.remove_vertices(toBeRemoved);
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

    // 1. Preprocessing
    auto size = graph.size();
    std::vector<vertex> selectedVertices;
    bool* visited = new bool[size];
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
        while (!nonContractedEnds.empty()) {
            vertex u = nonContractedEnds.back();    
            nonContractedEnds.pop_back();
            graph.add_edge(cc, u);
            graph.add_edge(u, cc);
        }
    }

    // 3. Remove vertices from each connected component
    for (auto v : selectedVertices) {
		vertexPositions2D[0].erase(vertexPositions2D[0].begin() + 2 * v, vertexPositions2D[0].begin() + 2 * v + 2);
		vertexPositions2D[1].erase(vertexPositions2D[1].begin() + 2 * v, vertexPositions2D[1].begin() + 2 * v + 2);
		vertexVelocities2D[0].erase(vertexVelocities2D[0].begin() + 2 * v, vertexVelocities2D[0].begin() + 2 * v + 2);
		vertexVelocities2D[1].erase(vertexVelocities2D[1].begin() + 2 * v, vertexVelocities2D[1].begin() + 2 * v + 2);
		vertexStates.erase(vertexStates.begin() + v);
    }
    graph.remove_vertices(selectedVertices);
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
    return vertexPositions2D[readBufferId];
}

const std::vector<unsigned int>& GraphManager::States() const {
    return vertexStates;
}

const core::Graph& GraphManager::Graph() const {
    return graph;
}

const std::vector<unsigned int> GraphManager::GetEdges() const {
    std::vector<unsigned int> edges;
    for (unsigned int i = 0; i < graph.size(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (graph.has_edge(i, j) || graph.has_edge(j, i)) {
                edges.push_back(i);
                edges.push_back(j);
            }
        }
    }
    return edges;
}

const std::pair<float, float> GraphManager::BoundingSize() {
    return std::make_pair(boundingWidth, boundingHeight);
}

const std::pair<float, float> GraphManager::Center() {
    return std::make_pair(centerX, centerY);
}

void GraphManager::OnDrag(float dx, float dy) {
    dragging = true;
    for (unsigned int i = 0; i < graph.size(); i++) {
        if (vertexStates[i] & 0b01u) {
            vertexPositions2D[readBufferId][2 * i] += dx;
            vertexPositions2D[readBufferId][2 * i + 1] += dy;
        }
    }
}

void GraphManager::OnDrop() {
    dragging = false;
}

void GraphManager::Stop() {
    for (unsigned int i = 0; i < graph.size(); i++) {
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
    }
}
