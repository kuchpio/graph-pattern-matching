#pragma once

#include "core.h"
#include <vector>

class GraphManager {
    core::Graph graph;

    unsigned int readBufferId = 0;
    std::vector<float> vertexPositions2D[2], vertexVelocities2D[2];
    std::vector<unsigned int> vertexStates;

    const float C[5] = {-2.0f, 0.1f, 0.2f, -0.01f, -10.0f};

  public:
    GraphManager();
    void Initialize(core::Graph&& graph);
    void UpdatePositions(float deltaTimeSeconds);
    void HandleClick();
    const std::vector<float>& Positions2D() const;
    const std::vector<unsigned int>& States() const;
    const core::Graph& Graph() const;
    const std::vector<unsigned int> GetEdges() const;
};
