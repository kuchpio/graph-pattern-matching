#pragma once

#include <thread>
#include "wx/wx.h"

#include "pattern.h"

#include "graphPanel.h"

class Frame : public wxFrame {
    GraphPanel *patternPanel, *searchSpacePanel;

    wxCheckBox* inducedCheckbox;
    wxRadioButton *subgraphRadioButton, *minorRadioButton, *topologicalMinorRadioButton;
    wxButton *startStopMatchingButton, *showMatchingButton;
    wxStaticText* matchingStatus;

    std::thread matcherThread;
    pattern::PatternMatcher* currentlyWorkingMatcher;
    bool isCloseRequested;

    void OnMatchingStart();
    void OnMatchingStop();
    void OnMatchingComplete(bool matchFound);
    void OnCloseRequest(wxCloseEvent& event);
    void ClearMatching();
    pattern::PatternMatcher* GetSelectedMatcher() const;

  public:
    Frame(const wxString& title);
};
