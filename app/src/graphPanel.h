#pragma once

#include <chrono>
#include "wx/wx.h"
#include "core.h"
#include "graphManager.h"

#include "graphCanvas.h"

class GraphPanel : public wxPanel {
    GraphCanvas* canvas{nullptr};
    wxButton* openButton;
    wxButton *addButton, *deleteButton, *connectButton, *disconnectButton, *contractButton, *subdivideButton;
    wxButton *undoButton, *redoButton;
    wxStaticText* fileInfoLabel;
    const std::function<void()> fileOpenCallback;

    GraphManager manager;
    using animationClock = std::chrono::high_resolution_clock;
    std::chrono::time_point<animationClock> lastFrameTime;

    void OnIdle(wxIdleEvent& event);
    void OpenFromFile(wxCommandEvent& event);
    void SaveToFile(wxCommandEvent& event);

  public:
    GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> fileOpenCallback);

    const core::Graph& GetGraph() const;
    void OnMatchingStart();
    void OnMatchingEnd();
};
