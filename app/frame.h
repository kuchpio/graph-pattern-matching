#pragma once

#include <thread>
#include "wx/wx.h"

#include "pattern.h"

class Frame : public wxFrame {
    wxCheckBox *inducedCheckbox;
    wxRadioButton *subgraphRadioButton, *minorRadioButton, *topologicalMinorRadioButton;
    wxButton* startMatchingButton;
    wxStaticText *matchingStatus;

    std::thread matcherThread;
    bool isMathing;

    std::unique_ptr<pattern::PatternMatcher> GetSelectedMatcher() const;
  public:
    Frame(const wxString& title);
};
