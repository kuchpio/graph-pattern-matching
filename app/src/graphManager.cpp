#include "graphManager.h"

GraphManager::GraphManager() : graph(0) {
}

void GraphManager::Initialize(core::Graph&& newGraph) {
    graph = newGraph;
    vertexPositions2D[0] = std::vector<float>(2 * graph.size());
    vertexPositions2D[1] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[0] = std::vector<float>(2 * graph.size());
    vertexVelocities2D[1] = std::vector<float>(2 * graph.size());
    vertexStates = std::vector<unsigned int>(2 * graph.size());

    for (unsigned int i = 0; i < graph.size(); i++) {
        vertexPositions2D[readBufferId][2 * i] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexPositions2D[readBufferId][2 * i + 1] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
        vertexStates[i] = 0;
    }
}

void GraphManager::UpdatePositions(float deltaTimeSeconds) {
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

        float v_x = vertexVelocities2D[readBufferId][2 * i];
        float v_y = vertexVelocities2D[readBufferId][2 * i + 1];

        float tractionCoefficient = C[4];
        a_x += tractionCoefficient * v_x;
        a_y += tractionCoefficient * v_y;

        vertexVelocities2D[1 - readBufferId][2 * i] = v_x + a_x * deltaTimeSeconds;
        vertexVelocities2D[1 - readBufferId][2 * i + 1] = v_y + a_y * deltaTimeSeconds;
        vertexPositions2D[1 - readBufferId][2 * i] = x + v_x * deltaTimeSeconds;
        vertexPositions2D[1 - readBufferId][2 * i + 1] = y + v_y * deltaTimeSeconds;
    }

    readBufferId = 1 - readBufferId;
}

void GraphManager::HandleClick() {
	std::size_t i = rand() % graph.size();
	vertexStates[i] = 1 - vertexStates[i];
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
