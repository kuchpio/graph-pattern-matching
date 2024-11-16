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

Frame::Frame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto mainPanel = new wxPanel(this);
    auto modeLabel = new wxStaticText(mainPanel, wxID_ANY, "Mode:");
    inducedCheckbox = new wxCheckBox(mainPanel, wxID_ANY, "Induced");
    subgraphRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Subgraph", wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    minorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Minor");
    topologicalMinorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Topological minor");
    startMatchingButton = new wxButton(mainPanel, wxID_ANY, "Match");
    auto showMatchingButton = new wxButton(mainPanel, wxID_ANY, "Show Matching");
    matchingStatus = new wxStaticText(mainPanel, wxID_ANY, "");
    auto optionsButton = new wxButton(mainPanel, wxID_ANY, "Settings");

    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    mainPanelSizer->Add(modeLabel, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(inducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(subgraphRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(minorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(topologicalMinorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(1);
    mainPanelSizer->Add(startMatchingButton, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(showMatchingButton, 0, wxALIGN_CENTER | wxLEFT, 5);
    mainPanelSizer->Add(matchingStatus, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(2);
    mainPanelSizer->Add(optionsButton, 0, wxALIGN_CENTER);
    auto mainPanelPaddedSizer = new wxBoxSizer(wxVERTICAL);
    mainPanelPaddedSizer->Add(mainPanelSizer, 0, wxALL | wxEXPAND, 5);
    mainPanel->SetSizerAndFit(mainPanelPaddedSizer);

    auto splitter = new wxSplitterWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_LIVE_UPDATE);

    auto patternPanel = new GraphPanel(splitter, "Pattern graph");
    auto searchSpacePanel = new GraphPanel(splitter, "Search space graph");

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(patternPanel, searchSpacePanel);

    sizer->Add(mainPanel, 0, wxEXPAND);
    sizer->Add(splitter, 1, wxEXPAND);

    this->SetSizerAndFit(sizer);

    isMathing = false;
    startMatchingButton->Bind(wxEVT_BUTTON, [this, patternPanel, searchSpacePanel](wxCommandEvent& event) {
        isMathing = true;
        startMatchingButton->Disable();
        auto matcher = GetSelectedMatcher();

        const core::Graph& patternGraph = patternPanel->GetGraph();
        const core::Graph& searchSpaceGraph = searchSpacePanel->GetGraph();

        matchingStatus->SetLabel("Matching...");

        matcherThread = std::thread([this](
            std::unique_ptr<pattern::PatternMatcher> matcher, 
            const core::Graph& patternGraph, 
            const core::Graph& searchSpaceGraph
            ) { 

            auto result = matcher->match(searchSpaceGraph, patternGraph);

            wxTheApp->CallAfter([this, result]() { 
                matcherThread.join();
                if (result) {
                    matchingStatus->SetLabel("Match found");
                } else {
                    matchingStatus->SetLabel("Match not found");
                }
                startMatchingButton->Enable();
                isMathing = false;
            });
        }, std::move(matcher), patternGraph, searchSpaceGraph);
    });

    this->Bind(wxEVT_CLOSE_WINDOW, [this](wxCloseEvent& e) { 
        if (isMathing) {
            e.Veto();
            matchingStatus->SetLabel("Waiting for the matching process to end");
        } else {
            this->Destroy();
        }
    });
}

std::unique_ptr<pattern::PatternMatcher> Frame::GetSelectedMatcher() const {

	if (subgraphRadioButton->GetValue()) {
		if (inducedCheckbox->GetValue()) {
            return std::unique_ptr<pattern::PatternMatcher>(new pattern::InducedSubgraphMatcher());
		}
        
        return std::unique_ptr<pattern::PatternMatcher>(new pattern::SubgraphMatcher());
	}

    if (minorRadioButton->GetValue()) {
		if (inducedCheckbox->GetValue()) {
            return std::unique_ptr<pattern::PatternMatcher>(new pattern::InducedMinorMatcher());
		}
        
        return std::unique_ptr<pattern::PatternMatcher>(new pattern::MinorMatcher());
	}

	if (inducedCheckbox->GetValue()) {
		return std::unique_ptr<pattern::PatternMatcher>(new pattern::TopologicalInducedMinorMatcher());
	}
	
    return std::unique_ptr<pattern::PatternMatcher>(new pattern::TopologicalMinorMatcher());
}
