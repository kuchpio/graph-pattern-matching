#pragma once

#include <thread>
#include <filesystem>
#include "wx/wx.h"
#include "wx/config.h"
#include "configDefaults.h"
#include "wx/dynlib.h"

#include "pattern.h"

#include "graphPanel.h"

class Frame : public wxFrame {
    const wxString APP_NAME_ID = "GraphPatternMatching";
    void LoadConfig(const wxConfig* config);

    GraphPanel *patternPanel, *searchSpacePanel;

    wxCheckBox* inducedCheckbox;
    wxRadioButton *subgraphRadioButton, *minorRadioButton, *topologicalMinorRadioButton;
    wxButton *startStopMatchingButton, *startStopCustomMatchingButton;
    wxStaticText* matchingStatus;
    wxString customAlgorithmName;
    wxDynamicLibrary plugin;

    std::unordered_map<std::string, int> selectedAlgorithm;
    std::thread matcherThread;
    core::IPatternMatcher* currentlyWorkingMatcher = nullptr;
    bool isCloseRequested = false, isMatchingAlgorithmBeingStopped = false;
    std::optional<std::vector<vertex>> matchingResult;

    void OnMatchingStart();
    void OnMatchingStop();
    void OnMatchingComplete();
    void OnCloseRequest(wxCloseEvent& event);
    void ClearMatching();
    void UpdateControlsState();
    core::IPatternMatcher* GetSelectedMatcher() const;
    core::IPatternMatcher* GetCustomMatcher() const;
    std::vector<std::optional<std::pair<float, float>>> GetPatternMatchingAlignment();
    std::vector<std::optional<std::pair<float, float>>> GetSearchSpaceMatchingAlignment();

  public:
    Frame(const wxString& title);
};
