#pragma once

#include "wx/dialog.h"
#include "wx/checkbox.h"
#include "wx/config.h"

class ConfigDialog : public wxDialog {
    wxCheckBox *animateContractionCheckbox, *animateAlignmentCheckbox;

  public:
    ConfigDialog(wxWindow* parent);
    void Load(wxConfigBase* config);
    void Save(wxConfigBase* config) const;
};