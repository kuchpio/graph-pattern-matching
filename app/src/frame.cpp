#include "wx/splitter.h"
#include "wx/app.h"
#include "wx/dynlib.h"
#include <numeric>

#include "solvers.h"

#include "frame.h"
#include "graphPanel.h"
#include "configDialog.h"

Frame::Frame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto mainPanel = new wxPanel(this);
    auto modeLabel = new wxStaticText(mainPanel, wxID_ANY, "Mode:");
    inducedCheckbox = new wxCheckBox(mainPanel, wxID_ANY, "Induced");
    subgraphRadioButton =
        new wxRadioButton(mainPanel, wxID_ANY, "Subgraph", wxDefaultPosition, wxDefaultSize, wxRB_GROUP);
    subgraphRadioButton->SetValue(true);
    minorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Minor");
    topologicalMinorRadioButton = new wxRadioButton(mainPanel, wxID_ANY, "Topological minor");
    startStopMatchingButton = new wxButton(mainPanel, wxID_ANY, "Match");
    startStopCustomMatchingButton = new wxButton(mainPanel, wxID_ANY, "Custom Match");
    matchingStatus = new wxStaticText(mainPanel, wxID_ANY, "");
    auto optionsButton = new wxButton(mainPanel, wxID_ANY, "Settings");
    auto configDialog = new ConfigDialog(this);

    auto mainPanelSizer = new wxBoxSizer(wxHORIZONTAL);
    mainPanelSizer->Add(modeLabel, 0, wxALIGN_CENTER | wxLEFT, 5);
    mainPanelSizer->Add(inducedCheckbox, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(subgraphRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(minorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(topologicalMinorRadioButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(startStopMatchingButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(startStopCustomMatchingButton, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->Add(matchingStatus, 0, wxALIGN_CENTER | wxLEFT, 10);
    mainPanelSizer->AddStretchSpacer(1);
    mainPanelSizer->Add(optionsButton, 0, wxALIGN_CENTER);
    auto mainPanelPaddedSizer = new wxBoxSizer(wxVERTICAL);
    mainPanelPaddedSizer->Add(mainPanelSizer, 0, wxALL | wxEXPAND, 5);
    mainPanel->SetSizerAndFit(mainPanelPaddedSizer);

    auto splitter = new wxSplitterWindow(this);

    patternPanel = new GraphPanel(
        splitter, "Pattern graph", [this]() { ClearMatching(); },
        [this]() {
            UpdateControlsState();
            if (isCloseRequested) Close();
        },
        [this]() { return GetPatternMatchingAlignment(); });
    searchSpacePanel = new GraphPanel(
        splitter, "Search space graph", [this]() { ClearMatching(); },
        [this]() {
            UpdateControlsState();
            if (isCloseRequested) Close();
        },
        [this]() { return GetSearchSpaceMatchingAlignment(); });

    splitter->SetSashGravity(0.5);
    splitter->SplitVertically(searchSpacePanel, patternPanel);

    sizer->Add(splitter, 1, wxEXPAND);
    sizer->Add(mainPanel, 0, wxEXPAND);

    this->SetSizerAndFit(sizer);

    startStopMatchingButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        if (currentlyWorkingMatcher) {
            OnMatchingStop();
        } else {
            currentlyWorkingMatcher = GetSelectedMatcher();
            startStopMatchingButton->SetLabel("Stop");
            startStopCustomMatchingButton->Disable();
            OnMatchingStart();
        }
    });
    startStopCustomMatchingButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        if (currentlyWorkingMatcher) {
            OnMatchingStop();
        } else {
            try {
                currentlyWorkingMatcher = GetCustomMatcher();
            } catch (const std::runtime_error& err) {
                wxMessageBox("Could not run `" + customAlgorithmName + "`\nError: " + std::string(err.what()));
                wxLogDebug("ERROR: %s", err.what());
                return;
            }
            startStopMatchingButton->Disable();
            startStopCustomMatchingButton->SetLabel("Stop");
            OnMatchingStart();
        }
    });
    optionsButton->Bind(wxEVT_BUTTON, [this, configDialog](wxCommandEvent& event) {
        auto config = new wxConfig(APP_NAME_ID);
        configDialog->Load(config);
        configDialog->CenterOnParent();
        if (configDialog->ShowModal() == wxID_OK) {
            configDialog->Save(config);
            LoadConfig(config);
        }

        delete config;
    });
    Bind(wxEVT_CLOSE_WINDOW, &Frame::OnCloseRequest, this);

    auto config = new wxConfig(APP_NAME_ID);
    LoadConfig(config);
    delete config;

    UpdateControlsState();
}

void Frame::LoadConfig(const wxConfig* config) {
    ConfigDefaults defaults;

    auto triangulateImage = config->ReadObject(defaults.TRIANGULATE_ID, defaults.TRIANGULATE);

    GraphDrawingSettings settings;
    settings.contractionAnimationTotalTimeSeconds =
        config->ReadObject(defaults.CONTRACTION_TIME_ID, defaults.CONTRACTION_TIME);
    settings.alignmentAnimationTotalTimeSeconds =
        config->ReadObject(defaults.ALIGNMENT_TIME_ID, defaults.ALIGNMENT_TIME);
    settings.springStrength = config->ReadObject(defaults.SPRING_STRENGTH_ID, defaults.SPRING_STRENGTH);
    settings.springLength = config->ReadObject(defaults.SPRING_LENGTH_ID, defaults.SPRING_LENGTH);
    settings.nodeRepulsion = config->ReadObject(defaults.NODE_REPULSION_ID, defaults.NODE_REPULSION);
    settings.nodeDrag = config->ReadObject(defaults.NODE_DRAG_ID, defaults.NODE_DRAG);

    selectedAlgorithm[defaults.SELECTED_SUBGRAPH_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_SUBGRAPH_ALGORITHM_ID, defaults.SELECTED_SUBGRAPH_ALGORITHM);
    selectedAlgorithm[defaults.SELECTED_INDUCED_SUBGRAPH_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_INDUCED_SUBGRAPH_ALGORITHM_ID, defaults.SELECTED_INDUCED_SUBGRAPH_ALGORITHM);
    selectedAlgorithm[defaults.SELECTED_MINOR_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_MINOR_ALGORITHM_ID, defaults.SELECTED_MINOR_ALGORITHM);
    selectedAlgorithm[defaults.SELECTED_INDUCED_MINOR_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_INDUCED_MINOR_ALGORITHM_ID, defaults.SELECTED_INDUCED_MINOR_ALGORITHM);
    selectedAlgorithm[defaults.SELECTED_TOPOLOGICAL_MINOR_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_TOPOLOGICAL_MINOR_ALGORITHM_ID, defaults.SELECTED_TOPOLOGICAL_MINOR_ALGORITHM);
    selectedAlgorithm[defaults.SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM_ID] =
        config->Read(defaults.SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM_ID,
                     defaults.SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM);

    auto customAlgorithmEnabled =
        config->ReadObject(defaults.ENABLE_EXTERNAL_ALGORITHM_ID, defaults.ENABLE_EXTERNAL_ALGORITHM);
    if (customAlgorithmEnabled) {
        auto customMatchingAlgorithmPath = std::filesystem::path(
            (const char*)config->Read(defaults.EXTERNAL_ALGORITHM_PATH_ID, defaults.EXTERNAL_ALGORITHM_PATH).mb_str());

        if (plugin.IsLoaded()) plugin.Unload();
        if (!plugin.Load(customMatchingAlgorithmPath.string())) {
            wxMessageBox("Could not load library `" + customMatchingAlgorithmPath.filename().string());
            customAlgorithmEnabled = false;
        }
    }
    customAlgorithmName = config->Read(defaults.EXTERNAL_ALGORITHM_NAME_ID, defaults.EXTERNAL_ALGORITHM_NAME);

    searchSpacePanel->UpdateDrawingSettings(settings);
    searchSpacePanel->UpdateImageTriangulationSetting(triangulateImage);
    patternPanel->UpdateDrawingSettings(settings);
    patternPanel->UpdateImageTriangulationSetting(triangulateImage);

    startStopCustomMatchingButton->Show(customAlgorithmEnabled);
    startStopCustomMatchingButton->SetLabel("Match (" + customAlgorithmName + ")");
    Layout();
}

void Frame::OnMatchingStart() {
    const core::Graph& patternGraph = patternPanel->Manager().Graph();
    const core::Graph& searchSpaceGraph = searchSpacePanel->Manager().Graph();

    matchingStatus->SetLabel("Matching...");
    patternPanel->OnMatchingStart();
    searchSpacePanel->OnMatchingStart();

    matcherThread = std::thread(
        [this](const core::Graph& patternGraph, const core::Graph& searchSpaceGraph) {
            auto result = currentlyWorkingMatcher->match(searchSpaceGraph, patternGraph);

            wxTheApp->CallAfter([this, result = std::move(result)]() {
                matcherThread.join();

                matchingResult = result;
                OnMatchingComplete();
            });
        },
        patternGraph, searchSpaceGraph);
}

void Frame::OnMatchingStop() {
    if (isMatchingAlgorithmBeingStopped) return;
    isMatchingAlgorithmBeingStopped = true;
    currentlyWorkingMatcher->interrupt();

    UpdateControlsState();
    matchingStatus->SetLabel("Stopping the matching process...");
}

void Frame::OnMatchingComplete() {
    delete currentlyWorkingMatcher;
    currentlyWorkingMatcher = nullptr;
    isMatchingAlgorithmBeingStopped = false;

    UpdateControlsState();
    startStopMatchingButton->SetLabel("Match");
    startStopCustomMatchingButton->SetLabel("Match (" + customAlgorithmName + ")");

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
    if (!currentlyWorkingMatcher && !searchSpacePanel->IsImageLoading() && !patternPanel->IsImageLoading()) {
        this->Destroy();
        return;
    }

    event.Veto();
    isCloseRequested = true;
    matchingStatus->SetLabel("Stopping... The application will close soon.");
    if (currentlyWorkingMatcher && !isMatchingAlgorithmBeingStopped) {
        isMatchingAlgorithmBeingStopped = true;
        currentlyWorkingMatcher->interrupt();
    }
    UpdateControlsState();
}

void Frame::ClearMatching() {
    if (!isCloseRequested) matchingStatus->SetLabel(wxEmptyString);

    patternPanel->OnMatchingEnd();
    searchSpacePanel->OnMatchingEnd();
}

void Frame::UpdateControlsState() {
    auto enable =
        !isMatchingAlgorithmBeingStopped && !searchSpacePanel->IsImageLoading() && !patternPanel->IsImageLoading();
    startStopMatchingButton->Enable(enable);
    startStopCustomMatchingButton->Enable(enable);
}

core::IPatternMatcher* Frame::GetSelectedMatcher() const {
    ConfigDefaults defaults;

    if (subgraphRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            auto selected = selectedAlgorithm.at(defaults.SELECTED_INDUCED_SUBGRAPH_ALGORITHM_ID);
            if (selected == 0) return new pattern::Vf2InducedSubgraphSolver();
            return new pattern::InducedSubgraphMatcher();
        }
        auto selected = selectedAlgorithm.at(defaults.SELECTED_SUBGRAPH_ALGORITHM_ID);
#ifdef CUDA_ENABLED
        if (selected == 0) return new pattern::CudaSubgraphMatcher();
        if (selected == 1) return new pattern::Vf2SubgraphSolver();
#else
        if (selected == 0) return new pattern::Vf2SubgraphSolver();
#endif
        return new pattern::NativeSubgraphMatcher();
    }

    if (minorRadioButton->GetValue()) {
        if (inducedCheckbox->GetValue()) {
            auto selected = selectedAlgorithm.at(defaults.SELECTED_INDUCED_MINOR_ALGORITHM_ID);
            if (selected == 0) return new pattern::InducedMinorHeuristic();
            return new pattern::InducedMinorMatcher();
        }
        auto selected = selectedAlgorithm.at(defaults.SELECTED_MINOR_ALGORITHM_ID);
        if (selected == 0) return new pattern::MinerMinorMatcher();
        return new pattern::NativeMinorMatcher();
    }

    if (inducedCheckbox->GetValue()) {
        auto selected = selectedAlgorithm.at(defaults.SELECTED_INDUCED_TOPOLOGICAL_MINOR_ALGORITHM_ID);
        if (selected == 0) return new pattern::InducedTopologicalMinorHeuristicSolver();
        return new pattern::TopologicalInducedMinorMatcher();
    }
    auto selected = selectedAlgorithm.at(defaults.SELECTED_TOPOLOGICAL_MINOR_ALGORITHM_ID);
    if (selected == 0) return new pattern::TopologicalMinorHeuristicSolver();
    return new pattern::TopologicalInducedMinorMatcher();
}

core::IPatternMatcher* Frame::GetCustomMatcher() const {
    bool success;
    auto getPatternMatcher = (core::IPatternMatcher * (*)()) plugin.GetSymbol("GetPatternMatcher", &success);
    if (!success) throw std::runtime_error("Unable to find symbol `GetPatternMatcher`");

    return getPatternMatcher();
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
