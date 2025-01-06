#pragma once

#include <chrono>
#include <thread>
#include "wx/wx.h"
#include "core.h"
#include "graphManager.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
    GraphCanvas* canvas{nullptr};
    wxButton* openButton;
    wxButton *deleteButton, *connectButton, *disconnectButton, *contractButton, *subdivideButton;
    wxButton *undoButton, *redoButton;
    wxStaticText *fileInfoLabel, *FPSInfoLabel;
    wxCheckBox* autoVertexPositioningCheckbox;
    wxButton* loadButton;
    wxTextCtrl* vertexCountInput;
    bool matching;
    const std::function<void()> clearMatchingCallback;
    std::string pathToImage;

    GraphManager manager;
    using animationClock = std::chrono::high_resolution_clock;
    std::chrono::time_point<animationClock> lastFrameTime;

    static const unsigned int FPS_ANALYSIS_COUNT = 50;
    float fpsArray[FPS_ANALYSIS_COUNT] = {0.0};
    unsigned int fpsIndex = 0;

    std::optional<wxPoint> prevMousePoint{};
    std::optional<wxPoint> areaSelectionStartPoint{};
    std::optional<vertex> connectionStartVertex{};
    bool vertexDragging = false;

    void OnIdle(wxIdleEvent& event);
    void OpenFromFile(wxCommandEvent& event);
    void SaveToFile(wxCommandEvent& event);
    void OnCanvasClick(wxMouseEvent& event);
    void OnCanvasMotion(wxMouseEvent& event);
    void OnGraphUpdate();

  public:
    GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> clearMatchingCallback);

    const core::Graph& GetGraph() const;
    void OnMatchingStart();
    void OnMatchingEnd(const std::vector<unsigned int>& labelling);
};
