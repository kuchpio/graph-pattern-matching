#pragma once

#include <chrono>
#include "wx/wx.h"
#include "core.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
    GraphCanvas* canvas{nullptr};
    wxButton* openButton;
    wxButton *addButton, *deleteButton, *connectButton, *disconnectButton, *contractButton, *subdivideButton;
    wxButton *undoButton, *redoButton;
    wxStaticText* fileInfoLabel;
    const std::function<void()> fileOpenCallback;

    core::Graph graph;

    unsigned int readBufferId = 0;
    float* vertexPositions2D[2] = {nullptr, nullptr};
    float* vertexVelocities2D[2] = {nullptr, nullptr};
    unsigned int* vertexStates = nullptr;

    const float C[5] = {-2.0f, 0.1f, 0.2f, -0.01f, -10.0f};
    using animationClock = std::chrono::high_resolution_clock;
    std::chrono::time_point<animationClock> lastFrameTime;

    void InitGraphSimulation();
    void OnIdle(wxIdleEvent& event);
    void OpenFromFile(wxCommandEvent& event);
    void SaveToFile(wxCommandEvent& event);

  public:
    GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> fileOpenCallback);
    ~GraphPanel();

    const core::Graph& GetGraph() const;
    void OnMatchingStart();
    void OnMatchingEnd();
};
