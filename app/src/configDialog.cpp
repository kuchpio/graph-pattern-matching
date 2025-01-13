#include "configDialog.h"
#include "wx/sizer.h"
#include "wx/statline.h"
#include "configDefaults.h"

ConfigDialog::ConfigDialog(wxWindow* parent) : wxDialog(parent, wxID_ANY, "Settings") {
    auto animationConfigSizer = InitAnimationConfig();
    auto matchingConfigSizer = InitMatchingConfig();
    auto externalAlgorithmConfigSizer = InitExternalAlgorithmConfig();
	auto bottomSizer = CreateStdDialogButtonSizer(wxOK | wxCANCEL | wxHELP);
    auto restoreDefaultsButton = bottomSizer->GetHelpButton();
    restoreDefaultsButton->SetLabel("Defaults");

    restoreDefaultsButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
		ConfigDefaults defaults;
		contractionAnimationTimeSlider->SetValue((int)10.0 * defaults.CONTRACTION_TIME);
		alignmentAnimationTimeSlider->SetValue((int)10.0 * defaults.ALIGNMENT_TIME);
		springStrengthSlider->SetValue((int)10.0 * defaults.SPRING_STRENGTH);
		springLengthSlider->SetValue((int)10.0 * defaults.SPRING_LENGTH);
		nodeRepulsionSlider->SetValue((int)10.0 * defaults.NODE_REPULSION);
		nodeDragSlider->SetValue((int)10.0 * defaults.NODE_DRAG);
		externalAlgorithmCheckbox->SetValue(defaults.ENABLE_EXTERNAL_ALGORITHM);
        externalAlgorithmPanel->Enable(externalAlgorithmCheckbox->IsChecked());
		externalAlgorithmNameTextbox->SetValue(defaults.EXTERNAL_ALGORITHM_NAME);
        externalAlgorithmLibraryPath = std::filesystem::path(defaults.EXTERNAL_ALGORITHM_PATH);
		externalAlgorithmLibraryPathTextbox->SetValue(externalAlgorithmLibraryPath.filename().string());
    });

    auto sizer = new wxBoxSizer(wxVERTICAL);
    sizer->Add(animationConfigSizer, 0, wxEXPAND | wxALL, 5);
    sizer->Add(matchingConfigSizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
    sizer->Add(externalAlgorithmConfigSizer, 0, wxEXPAND | wxALL, 5);
    sizer->Add(bottomSizer, 0, wxEXPAND | wxALL, 5);
    SetSizerAndFit(sizer);
}

wxSizer* ConfigDialog::InitAnimationConfig() {
    auto animationSizer = new wxStaticBoxSizer(wxVERTICAL, this, "Drawing");
    auto animationPanel = animationSizer->GetStaticBox();

    auto animationSlidersPanel = new wxPanel(animationPanel);
    auto animationSlidersGrid = new wxGridSizer(3, 4, 0, 0);

    auto contractionAnimationTimeLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Contraction time");
    contractionAnimationTimeSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 0, 0, 100);
    auto alignmentAnimationTimeLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Alignment time");
    alignmentAnimationTimeSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 0, 0, 100);
    auto springStrengthLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Spring strength");
    springStrengthSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 1, 0, 100);
    auto springLengthLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Spring length");
    springLengthSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 1, 0, 100);
    auto nodeRepulsionLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Node repulsion");
    nodeRepulsionSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 1, 0, 100);
    auto nodeDragLabel = new wxStaticText(animationSlidersPanel, wxID_ANY, "Node drag");
    nodeDragSlider = new wxSlider(animationSlidersPanel, wxID_ANY, 0, 0, 100);

    animationSlidersGrid->Add(contractionAnimationTimeLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(contractionAnimationTimeSlider, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(alignmentAnimationTimeLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(alignmentAnimationTimeSlider, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(springStrengthLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(springStrengthSlider, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(springLengthLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(springLengthSlider, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(nodeRepulsionLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(nodeRepulsionSlider, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(nodeDragLabel, 0, wxALIGN_CENTER);
    animationSlidersGrid->Add(nodeDragSlider, 0, wxALIGN_CENTER);
    animationSlidersPanel->SetSizerAndFit(animationSlidersGrid);

    animationSizer->Add(animationSlidersPanel);

    return animationSizer;
}

wxSizer* ConfigDialog::InitMatchingConfig() {
    auto matchingConfigSizer = new wxStaticBoxSizer(wxVERTICAL, this, "Matching");
    auto matchingConfigPanel = matchingConfigSizer->GetStaticBox();

    auto algorithmSelectionPanel = new wxPanel(matchingConfigPanel, wxID_ANY);
    auto algorithmLabel = new wxStaticText(algorithmSelectionPanel, wxID_ANY, "Algorithm: ");
    isInducedCheckbox = new wxCheckBox(algorithmSelectionPanel, wxID_ANY, "Induced");
    isSubgraphRadiobutton = new wxRadioButton(algorithmSelectionPanel, wxID_ANY, "Subgraph", wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    isMinorRadiobutton = new wxRadioButton(algorithmSelectionPanel, wxID_ANY, "Minor");
    isTopologicalMinorRadiobutton = new wxRadioButton(algorithmSelectionPanel, wxID_ANY, "Topological minor");
    auto algorithmSelectionSeparator = new wxStaticLine(matchingConfigPanel);
    wxString choices[] = {"A", "B"};
    algorithmChoice = new wxChoice(matchingConfigPanel, wxID_ANY, wxDefaultPosition, wxDefaultSize, 2, choices);
    auto algorithmSelectionSizer = new wxBoxSizer(wxHORIZONTAL);
    algorithmSelectionSizer->Add(algorithmLabel);
    algorithmSelectionSizer->Add(isInducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    algorithmSelectionSizer->Add(isSubgraphRadiobutton, 0, wxALIGN_CENTER | wxLEFT, 10);
    algorithmSelectionSizer->Add(isMinorRadiobutton, 0, wxALIGN_CENTER | wxLEFT, 10);
    algorithmSelectionSizer->Add(isTopologicalMinorRadiobutton, 0, wxALIGN_CENTER | wxLEFT, 10);
    algorithmSelectionPanel->SetSizerAndFit(algorithmSelectionSizer);

    matchingConfigSizer->Add(algorithmSelectionPanel, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
    matchingConfigSizer->Add(algorithmSelectionSeparator, 0, wxEXPAND | wxALL, 5);
    matchingConfigSizer->Add(algorithmChoice, 0, wxEXPAND | wxLEFT | wxRIGHT, 10);

    return matchingConfigSizer;
}

wxSizer* ConfigDialog::InitExternalAlgorithmConfig() {
    auto externalMatchingConfigSizer = new wxStaticBoxSizer(wxVERTICAL, this, "External matching algorithm");
    auto externalMatchingConfigPanel = externalMatchingConfigSizer->GetStaticBox();

    externalAlgorithmCheckbox = new wxCheckBox(externalMatchingConfigPanel, wxID_ANY, "Enabled");

    auto externalAlgorithmSelectionSeparator = new wxStaticLine(externalMatchingConfigPanel);

    externalAlgorithmPanel = new wxPanel(externalMatchingConfigPanel);
    auto externalAlgorithmSizer = new wxBoxSizer(wxHORIZONTAL);
    auto externalAlgorithmNameLabel = new wxStaticText(externalAlgorithmPanel, wxID_ANY, "Name: ");
    externalAlgorithmNameTextbox = new wxTextCtrl(externalAlgorithmPanel, wxID_ANY);
    auto externalAlgorithmLibraryLabel = new wxStaticText(externalAlgorithmPanel, wxID_ANY, "Library: ");
    externalAlgorithmLibraryPathTextbox = new wxTextCtrl(externalAlgorithmPanel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY);
    auto externalAlgorithmLibrarySelectButton = new wxButton(externalAlgorithmPanel, wxID_ANY, "Select");
    externalAlgorithmSizer->AddSpacer(10);
    externalAlgorithmSizer->Add(externalAlgorithmNameLabel, 0, wxALIGN_CENTER | wxRIGHT, 5);
    externalAlgorithmSizer->Add(externalAlgorithmNameTextbox, 0, wxALIGN_CENTER);
    externalAlgorithmSizer->AddSpacer(20);
    externalAlgorithmSizer->Add(externalAlgorithmLibraryLabel, 0, wxALIGN_CENTER | wxRIGHT, 5);
    externalAlgorithmSizer->Add(externalAlgorithmLibraryPathTextbox, 0, wxALIGN_CENTER | wxRIGHT, 5);
    externalAlgorithmSizer->Add(externalAlgorithmLibrarySelectButton, 0, wxALIGN_CENTER);
    externalAlgorithmSizer->AddSpacer(10);
    externalAlgorithmPanel->SetSizerAndFit(externalAlgorithmSizer);

    externalMatchingConfigSizer->Add(externalAlgorithmCheckbox, 0, wxLEFT, 5);
    externalMatchingConfigSizer->Add(externalAlgorithmSelectionSeparator, 0, wxEXPAND | wxALL, 5);
    externalMatchingConfigSizer->Add(externalAlgorithmPanel, 0, wxEXPAND, 5);

    externalAlgorithmCheckbox->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& event) {
        externalAlgorithmPanel->Enable(externalAlgorithmCheckbox->IsChecked());
    });
    externalAlgorithmLibrarySelectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
		auto fileDialog = new wxFileDialog(this, "Choose a file to open", wxEmptyString, wxEmptyString,
							 "Libraries (*.so; *.dll)|*.so;*.dll", wxFD_OPEN);

		if (fileDialog->ShowModal() != wxID_OK) {
			fileDialog->Destroy();
			return;
		}

        externalAlgorithmLibraryPath = std::filesystem::path((const char*)fileDialog->GetPath().mb_str());
        externalAlgorithmLibraryPathTextbox->SetValue(externalAlgorithmLibraryPath.filename().string());

		fileDialog->Destroy();
    });

    return externalMatchingConfigSizer;
}

void ConfigDialog::Load(wxConfigBase* config) {
    ConfigDefaults defaults;
    contractionAnimationTimeSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.CONTRACTION_TIME_ID, defaults.CONTRACTION_TIME));
    alignmentAnimationTimeSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.ALIGNMENT_TIME_ID, defaults.ALIGNMENT_TIME));
    springStrengthSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.SPRING_STRENGTH_ID, defaults.SPRING_STRENGTH));
    springLengthSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.SPRING_LENGTH_ID, defaults.SPRING_LENGTH));
    nodeRepulsionSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.NODE_REPULSION_ID, defaults.NODE_REPULSION));
    nodeDragSlider->SetValue(
        (int)10.0 * config->ReadObject(defaults.NODE_DRAG_ID, defaults.NODE_DRAG));
    externalAlgorithmCheckbox->SetValue(
        config->ReadObject(defaults.ENABLE_EXTERNAL_ALGORITHM_ID, defaults.ENABLE_EXTERNAL_ALGORITHM));
    externalAlgorithmPanel->Enable(externalAlgorithmCheckbox->IsChecked());
    externalAlgorithmNameTextbox->SetValue(
        config->Read(defaults.EXTERNAL_ALGORITHM_NAME_ID, defaults.EXTERNAL_ALGORITHM_NAME));
    externalAlgorithmLibraryPath = std::filesystem::path((const char*)
        config->Read(defaults.EXTERNAL_ALGORITHM_PATH_ID, defaults.EXTERNAL_ALGORITHM_PATH).mb_str());
    externalAlgorithmLibraryPathTextbox->SetValue(externalAlgorithmLibraryPath.filename().string());
}

void ConfigDialog::Save(wxConfigBase* config) const {
    ConfigDefaults defaults;
    config->Write(defaults.CONTRACTION_TIME_ID, contractionAnimationTimeSlider->GetValue() / 10.0f);
    config->Write(defaults.ALIGNMENT_TIME_ID, alignmentAnimationTimeSlider->GetValue() / 10.0f);
    config->Write(defaults.SPRING_STRENGTH_ID, springStrengthSlider->GetValue() / 10.0f);
    config->Write(defaults.SPRING_LENGTH_ID, springLengthSlider->GetValue() / 10.0f);
    config->Write(defaults.NODE_REPULSION_ID, nodeRepulsionSlider->GetValue() / 10.0f);
    config->Write(defaults.NODE_DRAG_ID, nodeDragSlider->GetValue() / 10.0f);
    config->Write(defaults.ENABLE_EXTERNAL_ALGORITHM_ID, externalAlgorithmCheckbox->IsChecked());
    config->Write(defaults.EXTERNAL_ALGORITHM_NAME_ID, externalAlgorithmNameTextbox->GetValue());
    config->Write(defaults.EXTERNAL_ALGORITHM_PATH_ID, externalAlgorithmLibraryPath.c_str());
}
