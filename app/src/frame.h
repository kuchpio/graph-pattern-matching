#pragma once

#include <thread>
#include "wx/wx.h"
#include "wx/config.h"
#include "configDefaults.h"

#include "pattern.h"

#include "graphPanel.h"

class Frame : public wxFrame {
    const wxString APP_NAME_ID = "GraphPatternMatching";
    void LoadConfig(const wxConfig* config);

    GraphPanel *patternPanel, *searchSpacePanel;

    wxCheckBox* inducedCheckbox;
    wxRadioButton *subgraphRadioButton, *minorRadioButton, *topologicalMinorRadioButton;
    wxButton* startStopMatchingButton;
    wxStaticText* matchingStatus;

    std::thread matcherThread;
    pattern::PatternMatcher* currentlyWorkingMatcher;
    bool isCloseRequested;
    std::optional<std::vector<vertex>> matchingResult;

    void OnMatchingStart();
    void OnMatchingStop();
    void OnMatchingComplete();
    void OnCloseRequest(wxCloseEvent& event);
    void ClearMatching();
    pattern::PatternMatcher* GetSelectedMatcher() const;
    std::vector<std::optional<std::pair<float, float>>> GetPatternMatchingAlignment();
    std::vector<std::optional<std::pair<float, float>>> GetSearchSpaceMatchingAlignment();

  public:
    Frame(const wxString& title);
};
