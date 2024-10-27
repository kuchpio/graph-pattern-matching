#include "wx/splitter.h"

#include "frame.h"
#include "graphPanel.h"

Frame::Frame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto mainPanel = new wxPanel(this);
    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    auto startMatchingButton = new wxButton(mainPanel, wxID_ANY, "Szukaj");

    mainPanelSizer->Add(startMatchingButton, 0, wxALL | wxALIGN_CENTER, 10);

    mainPanel->SetSizerAndFit(mainPanelSizer);

    auto splitter = new wxSplitterWindow(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSP_LIVE_UPDATE);

    auto leftPanel = new GraphPanel(splitter);
    auto rightPanel = new GraphPanel(splitter);

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(leftPanel, rightPanel);

    sizer->Add(mainPanel, 0, wxEXPAND);
    sizer->Add(splitter, 1, wxEXPAND);

    this->SetSizerAndFit(sizer);
    this->SetMinSize(wxSize(800, 600));
}
