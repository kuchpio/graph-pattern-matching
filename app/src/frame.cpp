#include "miner_minor_matcher.hpp"
#include "native_subgraph_matcher.h"
#include "vf2_induced_subgraph_solver.hpp"
#include "vf2_subgraph_solver.hpp"
#include "cuda_subgraph_matcher.h"
#include "wx/splitter.h"
#include "wx/app.h"
#include <numeric>

#include "subgraph_matcher.h"
#include "induced_subgraph_matcher.h"
#include "minor_matcher.h"
#include "induced_minor_matcher.h"
#include "topological_minor_heuristic_solver.h"
#include "topological_induced_minor_heuristic_solver.h"
#include "topological_induced_minor_matcher.h"
#include "induced_minor_heuristic.h"

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
    matchingStatus = new wxStaticText(mainPanel, wxID_ANY, "");
    auto optionsButton = new wxButton(mainPanel, wxID_ANY, "Settings");

    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    mainPanelSizer->Add(modeLabel, 0, wxALIGN_CENTER | wxLEFT, 5);
    mainPanelSizer->Add(inducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(subgraphRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(minorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(topologicalMinorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(startStopMatchingButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(matchingStatus, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(1);
    mainPanelSizer->Add(optionsButton, 0, wxALIGN_CENTER);
    auto mainPanelPaddedSizer = new wxBoxSizer(wxVERTICAL);
    mainPanelPaddedSizer->Add(mainPanelSizer, 0, wxALL | wxEXPAND, 5);
    mainPanel->SetSizerAndFit(mainPanelPaddedSizer);

    auto splitter = new wxSplitterWindow(this);

    patternPanel = new GraphPanel(splitter, "Pattern graph", 
        [this]() { ClearMatching(); }, [this]() { return GetPatternMatchingAlignment(); });
    searchSpacePanel = new GraphPanel(splitter, "Search space graph", 
        [this]() { ClearMatching(); }, [this]() { return GetSearchSpaceMatchingAlignment(); });

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(searchSpacePanel, patternPanel);

    sizer->Add(splitter, 1, wxEXPAND);
    sizer->Add(mainPanel, 0, wxEXPAND);

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

    const core::Graph& patternGraph = patternPanel->Manager().Graph();
    const core::Graph& searchSpaceGraph = searchSpacePanel->Manager().Graph();

    matchingStatus->SetLabel("Matching...");
    patternPanel->OnMatchingStart();
    searchSpacePanel->OnMatchingStart();

    matcherThread = std::thread(
        [this](const core::Graph& patternGraph, const core::Graph& searchSpaceGraph) {
            auto result = currentlyWorkingMatcher->match(searchSpaceGraph, patternGraph);

            wxTheApp->CallAfter([this, result = move(result)]() {
                matcherThread.join();

                matchingResult = result;
                OnMatchingComplete();
            });
        },
        patternGraph, searchSpaceGraph);
}

void Frame::OnMatchingStop() {
    startStopMatchingButton->Disable();
    matchingStatus->SetLabel("Stopping the matching process...");
    currentlyWorkingMatcher->interrupt();
}

void Frame::OnMatchingComplete() {
    delete currentlyWorkingMatcher;
    currentlyWorkingMatcher = nullptr;
    startStopMatchingButton->SetLabel("Match");
    startStopMatchingButton->Enable();

    if (matchingResult.has_value()) {
		std::vector<unsigned int> patternLabelling(patternPanel->Manager().Graph().size(), 0);
		std::vector<unsigned int> searchSpaceLabelling(searchSpacePanel->Manager().Graph().size(), 0);

        matchingStatus->SetLabel("Match found");

        std::iota(patternLabelling.begin(), patternLabelling.end(), 1);
        for (unsigned int v = 0; v < matchingResult.value().size(); v++) {
            auto u = matchingResult.value()[v];
            if (u < patternLabelling.size()) {
                searchSpaceLabelling[v] = patternLabelling[u];
            }
        }
		patternPanel->OnMatchingEnd(patternLabelling);
		searchSpacePanel->OnMatchingEnd(searchSpaceLabelling);
    } else {
        matchingStatus->SetLabel("Match not found");
		patternPanel->OnMatchingEnd();
		searchSpacePanel->OnMatchingEnd();
    }

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
    matchingStatus->SetLabel(wxEmptyString);

	patternPanel->OnMatchingEnd();
	searchSpacePanel->OnMatchingEnd();
}

pattern::PatternMatcher* Frame::GetSelectedMatcher() const {

    if (subgraphRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            return new pattern::Vf2InducedSubgraphSolver();
        }

        return new pattern::CudaSubgraphMatcher();
    }

    if (minorRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            return new pattern::InducedMinorHeuristic();
        }

        return new pattern::MinerMinorMatcher();
    }

    if (inducedCheckbox->GetValue()) {
        return new pattern::InducedTopologicalMinorHeuristicSolver();
    }

    return new pattern::TopologicalMinorHeuristicSolver();
}

std::vector<std::optional<std::pair<float, float>>> Frame::GetPatternMatchingAlignment() {
    auto patternSize = patternPanel->Manager().Graph().size();
    std::vector<std::optional<std::pair<float, float>>> alignment(patternSize);
    if (!matchingResult.has_value()) return alignment;

    auto& searchSpacePositions = searchSpacePanel->Manager().Positions2D();

    for (unsigned int v = 0; v < matchingResult.value().size(); v++) {
        auto u = matchingResult.value()[v];
        if (u >= patternSize) continue;
        alignment[u] = std::make_pair(searchSpacePositions[2 * v], searchSpacePositions[2 * v + 1]);
    }

    return alignment;
}

std::vector<std::optional<std::pair<float, float>>> Frame::GetSearchSpaceMatchingAlignment() {
    auto patternSize = patternPanel->Manager().Graph().size();
    auto searchSpaceSize = searchSpacePanel->Manager().Graph().size();
    std::vector<std::optional<std::pair<float, float>>> alignment(searchSpaceSize);
    if (!matchingResult.has_value()) return alignment;

    auto& patternPositions = patternPanel->Manager().Positions2D();

    for (unsigned int v = 0; v < matchingResult.value().size(); v++) {
        auto u = matchingResult.value()[v];
        if (u >= patternSize) continue;
        alignment[v] = std::make_pair(patternPositions[2 * u], patternPositions[2 * u + 1]);
    }

    return alignment;
}
