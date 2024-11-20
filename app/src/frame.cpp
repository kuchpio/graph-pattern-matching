#include "miner_minor_matcher.hpp"
#include "native_subgraph_matcher.h"
#include "vf2_induced_subgraph_solver.hpp"
#include "vf2_subgraph_solver.hpp"
#include "wx/splitter.h"
#include "wx/app.h"

#include "subgraph_matcher.h"
#include "induced_subgraph_matcher.h"
#include "minor_matcher.h"
#include "induced_minor_matcher.h"
#include "topological_minor_matcher.h"
#include "topological_induced_minor_matcher.h"

#include "frame.h"
#include "graphPanel.h"

Frame::Frame(const wxString& title)
    : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize), currentlyWorkingMatcher(nullptr),
      isCloseRequested(false) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto mainPanel = new wxPanel(this);
    auto modeLabel = new wxStaticText(mainPanel, wxID_ANY, "Mode:");
    inducedCheckbox = new wxCheckBox(mainPanel, wxID_ANY, "Induced");
    subgraphRadioButton =
        new wxRadioButton(mainPanel, wxID_ANY, "Subgraph", wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    minorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Minor");
    topologicalMinorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Topological minor");
    startStopMatchingButton = new wxButton(mainPanel, wxID_ANY, "Match");
    showMatchingButton = new wxButton(mainPanel, wxID_ANY, "Show Matching");
    showMatchingButton->Disable();
    matchingStatus = new wxStaticText(mainPanel, wxID_ANY, "");
    auto optionsButton = new wxButton(mainPanel, wxID_ANY, "Settings");

    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    mainPanelSizer->Add(modeLabel, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(inducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(subgraphRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(minorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(topologicalMinorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(1);
    mainPanelSizer->Add(startStopMatchingButton, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(showMatchingButton, 0, wxALIGN_CENTER | wxLEFT, 5);
    mainPanelSizer->Add(matchingStatus, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(2);
    mainPanelSizer->Add(optionsButton, 0, wxALIGN_CENTER);
    auto mainPanelPaddedSizer = new wxBoxSizer(wxVERTICAL);
    mainPanelPaddedSizer->Add(mainPanelSizer, 0, wxALL | wxEXPAND, 5);
    mainPanel->SetSizerAndFit(mainPanelPaddedSizer);

    auto splitter = new wxSplitterWindow(this);

    patternPanel = new GraphPanel(splitter, "Pattern graph", [this]() { ClearMatching(); });
    searchSpacePanel = new GraphPanel(splitter, "Search space graph", [this]() { ClearMatching(); });

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(searchSpacePanel, patternPanel);

    sizer->Add(mainPanel, 0, wxEXPAND);
    sizer->Add(splitter, 1, wxEXPAND);

    this->SetSizerAndFit(sizer);

    startStopMatchingButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        if (currentlyWorkingMatcher) {
            OnMatchingStop();
        } else {
            OnMatchingStart();
        }
    });
    Bind(wxEVT_CLOSE_WINDOW, &Frame::OnCloseRequest, this);
}

void Frame::OnMatchingStart() {
    currentlyWorkingMatcher = GetSelectedMatcher();
    startStopMatchingButton->SetLabel("Stop");

    const core::Graph& patternGraph = patternPanel->GetGraph();
    const core::Graph& searchSpaceGraph = searchSpacePanel->GetGraph();

    matchingStatus->SetLabel("Matching...");
    patternPanel->OnMatchingStart();
    searchSpacePanel->OnMatchingStart();

    matcherThread = std::thread(
        [this](const core::Graph& patternGraph, const core::Graph& searchSpaceGraph) {
            auto result = currentlyWorkingMatcher->match(searchSpaceGraph, patternGraph);

            wxTheApp->CallAfter([this, result]() {
                matcherThread.join();

                OnMatchingComplete(result.has_value());
            });
        },
        patternGraph, searchSpaceGraph);
}

void Frame::OnMatchingStop() {
    startStopMatchingButton->Disable();
    matchingStatus->SetLabel("Stopping the matching process...");

    // TODO: currentlyWorkingMatcher->cancel();
}

void Frame::OnMatchingComplete(bool matchFound) {
    delete currentlyWorkingMatcher;
    currentlyWorkingMatcher = nullptr;
    startStopMatchingButton->SetLabel("Match");
    startStopMatchingButton->Enable();
    if (matchFound) {
        matchingStatus->SetLabel("Match found");
        showMatchingButton->Enable();
    } else {
        matchingStatus->SetLabel("Match not found");
        showMatchingButton->Disable();
    }
    patternPanel->OnMatchingEnd();
    searchSpacePanel->OnMatchingEnd();

    if (isCloseRequested) Close();
}

void Frame::OnCloseRequest(wxCloseEvent& event) {
    if (currentlyWorkingMatcher) {
        event.Veto();
        OnMatchingStop();
        isCloseRequested = true;
    } else {
        this->Destroy();
    }
}

void Frame::ClearMatching() {
    showMatchingButton->Disable();
    matchingStatus->SetLabel(wxEmptyString);
}

pattern::PatternMatcher* Frame::GetSelectedMatcher() const {

    if (subgraphRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            return new pattern::Vf2InducedSubgraphSolver();
        }

        return new pattern::Vf2SubgraphSolver();
    }

    if (minorRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            return new pattern::InducedMinorMatcher();
        }

        return new pattern::MinerMinorMatcher();
    }

    if (inducedCheckbox->GetValue()) {
        return new pattern::TopologicalInducedMinorMatcher();
    }

    return new pattern::TopologicalMinorMatcher();
}
