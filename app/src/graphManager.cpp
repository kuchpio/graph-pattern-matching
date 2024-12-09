#include "graphManager.h"

GraphManager::GraphManager() : graph(0), boundingWidth(0), boundingHeight(0), centerX(0), centerY(0) {
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
}

void GraphManager::UpdatePositions(float deltaTimeSeconds) {
    float minX = FLT_MAX, minY = FLT_MAX, maxX = FLT_MIN, maxY = FLT_MIN;

    for (unsigned int i = 0; i < graph.size(); i++) {
        float x = vertexPositions2D[readBufferId][2 * i];
        float y = vertexPositions2D[readBufferId][2 * i + 1];

        float distOrigin = sqrtf(x * x + y * y);
        float gravityCoefficient = distOrigin < 0.1f ? 0.0f : C[3] / (distOrigin * distOrigin * distOrigin);

        float a_x = gravityCoefficient * x;
        float a_y = gravityCoefficient * y;

        for (unsigned int j = 0; j < graph.size(); j++) {
            if (i == j) continue;
            float dx = x - vertexPositions2D[readBufferId][2 * j];
            float dy = y - vertexPositions2D[readBufferId][2 * j + 1];
            float dist = sqrtf(dx * dx + dy * dy);

            float springCoefficient =
                !graph.has_edge(i, j) || !graph.has_edge(j, i) || dist < 0.01f ? 0.0f : C[0] * logf(dist / C[1]) / dist;
            float repelCoefficient = dist < 0.0001f ? 0.0f : C[2] / (dist * dist * dist);

            a_x += (repelCoefficient + springCoefficient) * x;
            a_y += (repelCoefficient + springCoefficient) * y;
        }

        float v_x = vertexStates[i] & 0b10u ? 0.0 : vertexVelocities2D[readBufferId][2 * i];
        float v_y = vertexStates[i] & 0b10u ? 0.0 : vertexVelocities2D[readBufferId][2 * i + 1];

        float tractionCoefficient = C[4];
        a_x += tractionCoefficient * v_x;
        a_y += tractionCoefficient * v_y;

        vertexVelocities2D[1 - readBufferId][2 * i] = v_x + a_x * deltaTimeSeconds;
        vertexVelocities2D[1 - readBufferId][2 * i + 1] = v_y + a_y * deltaTimeSeconds;
        float newX = vertexPositions2D[1 - readBufferId][2 * i] = x + v_x * deltaTimeSeconds;
        float newY = vertexPositions2D[1 - readBufferId][2 * i + 1] = y + v_y * deltaTimeSeconds;
        
        if (newX < minX) minX = newX;
        if (newX > maxX) maxX = newX;
        if (newY < minY) minY = newY;
        if (newY > maxY) maxY = newY;
    }

    readBufferId = 1 - readBufferId;
    boundingWidth = maxX - minX;
    boundingHeight = maxY - minY;
    centerX = (minX + maxX) / 2;
    centerY = (minY + maxY) / 2;
}

void GraphManager::HandleClick(float x, float y, float nodeRadius, bool isCtrl) {
    for (unsigned int i = 0; i < graph.size(); i++) {
		float dx = x - vertexPositions2D[readBufferId][2 * i];
		float dy = y - vertexPositions2D[readBufferId][2 * i + 1];
        if (!isCtrl) vertexStates[i] &= 0b10u;
        if (dx * dx + dy * dy <= nodeRadius * nodeRadius)
            vertexStates[i] ^= 0b01u;
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
            if (graph.has_edge(i, j)) {
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
