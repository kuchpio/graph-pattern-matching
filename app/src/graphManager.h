#pragma once

#include "core.h"
#include <vector>
#include "graphDrawingSettings.h"

class GraphManager {
    const float BOUNDS_MOVING_SPEED = 2.0;
    const float MIN_NODE_DISTANCE = 0.1f;

    core::Graph graph;
    GraphDrawingSettings settings;

    unsigned int readBufferId = 0;
    std::vector<float> vertexPositions2D[2], vertexVelocities2D[2];
    std::vector<float> vertexRenderedPositions2D;
    std::optional<float> animationTimeLeftSeconds;
    std::vector<vertex> tracking;
    vertex renderedVertexCount = 0;
    float boundingWidth, boundingHeight, centerX, centerY;
    std::vector<unsigned int> vertexStates;

    void ResizeAnimationData();
    static float Approach(float value, float goal, float change);

  public:
    GraphManager();

    void Initialize(core::Graph&& graph);
    void Initialize(core::Graph&& graph, std::vector<std::pair<float, float>>&& vertexPositions);
    void UpdatePositions(float deltaTimeSeconds, bool dragging);
    bool UpdateRenderedPositions(float deltaTimeSeconds);
    void UpdateBounds(float deltaTimeSeconds);
    bool IsAnimationRunning() const;
    vertex RenderedVertexCount() const;

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
    void AlignNodes(std::vector<std::optional<std::pair<float, float>>>& positions2D);

    const std::vector<float>& Positions2D() const;
    const std::pair<float, float> Position2D(vertex v) const;
    const std::vector<unsigned int> GetStates() const;
    const std::vector<unsigned int> GetEdges() const;
    const std::vector<unsigned int> GetRenderedLabelling(const std::vector<unsigned int>& labelling) const;
    const std::pair<float, float> BoundingSize();
    const std::pair<float, float> Center();
    const core::Graph& Graph() const;

    void UpdateSettings(GraphDrawingSettings settings);
};
