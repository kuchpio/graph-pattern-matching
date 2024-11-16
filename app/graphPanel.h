#pragma once

#include <chrono>
#include "wx/wx.h"
#include "core.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
  public:
    GraphPanel(wxWindow* parent, const wxString& title);
    ~GraphPanel();

  private:
    GraphCanvas* canvas{nullptr};

    core::Graph graph;
    unsigned int readBufferId = 0;
    float* vertexPositions2D[2] = {nullptr, nullptr};
    float* vertexVelocities2D[2] = {nullptr, nullptr};

    const float C[5] = {-2.0f, 0.1f, 0.2f, -0.01f, -10.0f};
    using animationClock = std::chrono::high_resolution_clock;
    std::chrono::time_point<animationClock> lastFrameTime;

    void InitGraphSimulation();
    void OnIdle(wxIdleEvent& event);
};
