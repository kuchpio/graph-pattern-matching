#pragma once

#include "wx/dialog.h"
#include "wx/checkbox.h"
#include "wx/radiobut.h"
#include "wx/choice.h"
#include "wx/config.h"
#include "wx/slider.h"
#include "wx/textctrl.h"
#include <filesystem>

class ConfigDialog : public wxDialog {
    wxCheckBox *isInducedCheckbox, *externalAlgorithmCheckbox;
    wxRadioButton *isSubgraphRadiobutton, *isMinorRadiobutton, *isTopologicalMinorRadiobutton;
    wxChoice* algorithmChoice;
    wxSlider *contractionAnimationTimeSlider, *alignmentAnimationTimeSlider, *springStrengthSlider, *springLengthSlider,
        *nodeRepulsionSlider, *nodeDragSlider;
    wxTextCtrl *externalAlgorithmNameTextbox, *externalAlgorithmLibraryPathTextbox;
    wxPanel* externalAlgorithmPanel;
    std::filesystem::path externalAlgorithmLibraryPath;

    wxSizer* InitAnimationConfig();
    wxSizer* InitMatchingConfig();
    wxSizer* InitExternalAlgorithmConfig();

  public:
    ConfigDialog(wxWindow* parent);
    void Load(wxConfigBase* config);
    void Save(wxConfigBase* config) const;
};