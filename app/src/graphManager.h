#pragma once

#include "core.h"
#include <vector>

class GraphManager {
    core::Graph graph;

    unsigned int readBufferId = 0;
    std::vector<float> vertexPositions2D[2], vertexVelocities2D[2];
    float boundingWidth, boundingHeight, centerX, centerY;
    std::vector<unsigned int> vertexStates;

    const float SPRING_STRENGTH = 5.0f;
    const float SPRING_LENGTH = 1.0f;
    const float REPULSION_STRENGTH = 10.0f;
    const float DRAG = 4.0f;

  public:
    GraphManager();

    void Initialize(core::Graph&& graph);
    void Initialize(core::Graph&& graph, std::vector<std::pair<float, float>>&& vertexPositions);
    void UpdatePositions(float deltaTimeSeconds, bool dragging);

    void Stop();

    bool AddVertex(float x, float y);
    void ClearSelection();
    std::vector<vertex> GetCollidingNodes(float x, float y, float nodeRadius) const;
    std::vector<vertex> GetCollidingNodes(float startX, float startY, float endX, float endY) const;
    void SelectNodes(const std::vector<vertex>& nodes);
    void ToggleNodes(const std::vector<vertex>& nodes);
    void ConnectNodes(vertex start, vertex end);
    void OnVertexDrag(float dx, float dy);
    void DeleteSelection();
    void ConnectSelection();
    void DisconnectSelection();
    void ContractSelection();
    void SubdivideSelection();
    void AnchorSelection();
    void FreeSelection();
    void UpdateBounds();

    const std::vector<float>& Positions2D() const;
    const std::pair<float, float> Position2D(vertex v) const;
    const std::vector<unsigned int>& States() const;
    const core::Graph& Graph() const;
    const std::vector<unsigned int> GetEdges() const;
    const std::pair<float, float> BoundingSize();
    const std::pair<float, float> Center();
};
