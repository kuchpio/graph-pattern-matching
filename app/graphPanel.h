#pragma once

#include <chrono>
#include "wx/wx.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
  public:
    GraphPanel(wxWindow* parent);
    ~GraphPanel();

  private:
    GraphCanvas* canvas{nullptr};

    unsigned int vertexCount = 0;
    bool* adjecencyMatrix = nullptr;
    unsigned int readBufferId = 0;
    float* vertexPositions2D[2] = {nullptr, nullptr};
    float* vertexVelocities2D[2] = {nullptr, nullptr};

    const float C[5] = {-2.0f, 0.1f, 0.2f, -0.01f, -10.0f};
    using animationClock = std::chrono::high_resolution_clock;
    std::chrono::time_point<animationClock> lastFrameTime;

    void InitRandomGraph(unsigned int vertexCount);
    void OnIdle(wxIdleEvent& event);
};
