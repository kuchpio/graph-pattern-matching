#pragma once

#include "core.h"
#include <vector>

class GraphManager {
    core::Graph graph;

    unsigned int readBufferId = 0;
    std::vector<float> vertexPositions2D[2], vertexVelocities2D[2];
    float boundingWidth, boundingHeight, centerX, centerY;
    std::vector<unsigned int> vertexStates;
    bool dragging;

    const float SPRING_STRENGTH = 5.0f;
    const float SPRING_LENGTH = 1.0f;
    const float REPULSION_STRENGTH = 10.0f;
    const float DRAG = 4.0f;

    void AddVertex(float x, float y);

  public:
    GraphManager();

    void Initialize(core::Graph&& graph);
    void UpdatePositions(float deltaTimeSeconds);

    void HandleClick(float x, float y, float nodeRadius, bool isCtrl, bool isDouble);
    void OnDrag(float dx, float dy);
    void OnDrop();
    void Stop();

    void ConnectSelection();
    void DisconnectSelection();
    void AnchorSelection();
    void FreeSelection();

    const std::vector<float>& Positions2D() const;
    const std::vector<unsigned int>& States() const;
    const core::Graph& Graph() const;
    const std::vector<unsigned int> GetEdges() const;
    const std::pair<float, float> BoundingSize();
    const std::pair<float, float> Center();
};
