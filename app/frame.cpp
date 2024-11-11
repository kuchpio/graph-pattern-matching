#include "wx/splitter.h"

#include "frame.h"
#include "graphPanel.h"

Frame::Frame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto mainPanel = new wxPanel(this);
    auto modeLabel = new wxStaticText(mainPanel, wxID_ANY, "Mode:");
    auto inducedCheckbox = new wxCheckBox(mainPanel, wxID_ANY, "Induced");
    auto subgraphRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Subgraph", wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    auto minorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Minor");
    auto topologicalMinorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Topological minor");
    auto startMatchingButton = new wxButton(mainPanel, wxID_ANY, "Match");
    auto showMatchingButton = new wxButton(mainPanel, wxID_ANY, "Show Matching");
    auto matchingStatus = new wxStaticText(mainPanel, wxID_ANY, "Matching...");
    auto optionsButton = new wxButton(mainPanel, wxID_ANY, "Settings");

    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    mainPanelSizer->Add(modeLabel, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(inducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(subgraphRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(minorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(topologicalMinorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(1);
    mainPanelSizer->Add(startMatchingButton, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(showMatchingButton, 0, wxALIGN_CENTER);
    mainPanelSizer->Add(matchingStatus, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(2);
    mainPanelSizer->Add(optionsButton, 0, wxALIGN_CENTER);
    auto mainPanelPaddedSizer = new wxBoxSizer(wxVERTICAL);
    mainPanelPaddedSizer->Add(mainPanelSizer, 0, wxALL | wxEXPAND, 5);
    mainPanel->SetSizerAndFit(mainPanelPaddedSizer);

    auto splitter = new wxSplitterWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_LIVE_UPDATE);

    auto leftPanel = new GraphPanel(splitter, "Pattern graph");
    auto rightPanel = new GraphPanel(splitter, "Search space graph");

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(leftPanel, rightPanel);

    sizer->Add(mainPanel, 0, wxEXPAND);
    sizer->Add(splitter, 1, wxEXPAND);

    this->SetSizerAndFit(sizer);
}
