#include "configDialog.h"
#include "wx/sizer.h"
#include "wx/checkbox.h"
#include "configDefaults.h"

ConfigDialog::ConfigDialog(wxWindow* parent) : wxDialog(parent, wxID_ANY, "Settings") {
    animateContractionCheckbox = new wxCheckBox(this, wxID_ANY, "Smooth contraction");
    animateAlignmentCheckbox = new wxCheckBox(this, wxID_ANY, "Smooth alignment");
    auto animationSizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "Animation");
    animationSizer->Add(animateContractionCheckbox);
    animationSizer->Add(animateAlignmentCheckbox);

	auto bottomSizer = CreateStdDialogButtonSizer(wxOK | wxCANCEL);

    auto sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(animationSizer);
    sizer->Add(bottomSizer);
    SetSizerAndFit(sizer);
}

void ConfigDialog::Load(wxConfigBase* config) {
    ConfigDefaults defaults;
    animateContractionCheckbox->SetValue(config->ReadBool(defaults.ANIMATE_CONTRACTION_ID, defaults.ANIMATE_CONTRACTION));
    animateAlignmentCheckbox->SetValue(config->ReadBool(defaults.ANIMATE_ALIGNMENT_ID, defaults.ANIMATE_ALIGNMENT));
}

void ConfigDialog::Save(wxConfigBase* config) const {
    ConfigDefaults defaults;
	config->Write(defaults.ANIMATE_CONTRACTION_ID, animateContractionCheckbox->IsChecked());
	config->Write(defaults.ANIMATE_ALIGNMENT_ID, animateAlignmentCheckbox->IsChecked());
}
