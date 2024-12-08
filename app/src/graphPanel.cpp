#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "wx/notebook.h"
#include "utils.h"

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> fileOpenCallback)
    : wxPanel(parent), fileOpenCallback(fileOpenCallback) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().Samplers(16).EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        canvas = new GraphCanvas(this, vAttrs);
        canvas->SetInitialSize(wxSize(0, 360));
        sizer->Add(canvas, 1, wxEXPAND);
    }

    auto nameLabel = new wxStaticText(this, wxID_ANY, title);

    auto notebook = new wxNotebook(this, wxID_ANY);

    auto filePanel = new wxPanel(notebook);
    auto fileSizer = new wxBoxSizer(wxHORIZONTAL);
    auto saveButton = new wxButton(filePanel, wxID_ANY, "Save");
    openButton = new wxButton(filePanel, wxID_ANY, "Open");
    fileInfoLabel = new wxStaticText(filePanel, wxID_ANY, "Open a file to load the graph.");
    auto vertexCountInput = new wxTextCtrl(filePanel, wxID_ANY);
    vertexCountInput->SetHint("Vertex count");
    vertexCountInput->SetMinSize(wxSize(120, wxDefaultCoord));
    vertexCountInput->Disable();
    auto loadButton = new wxButton(filePanel, wxID_ANY, "Load");
    loadButton->Disable();
    fileSizer->Add(saveButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(openButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(fileInfoLabel, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->AddStretchSpacer(1);
    fileSizer->Add(vertexCountInput, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(loadButton, 0, wxALIGN_CENTER | wxLEFT | wxRIGHT | wxTOP | wxBOTTOM, 5);
    filePanel->SetSizerAndFit(fileSizer);
    notebook->AddPage(filePanel, "File");

    auto modifyPanel = new wxPanel(notebook);
    auto modifySizer = new wxBoxSizer(wxHORIZONTAL);
    addButton = new wxButton(modifyPanel, wxID_ANY, "Add");
    deleteButton = new wxButton(modifyPanel, wxID_ANY, "Delete");
    connectButton = new wxButton(modifyPanel, wxID_ANY, "Connect");
    disconnectButton = new wxButton(modifyPanel, wxID_ANY, "Disconnect");
    contractButton = new wxButton(modifyPanel, wxID_ANY, "Contract");
    subdivideButton = new wxButton(modifyPanel, wxID_ANY, "Subdivide");
    undoButton = new wxButton(modifyPanel, wxID_ANY, "Undo");
    redoButton = new wxButton(modifyPanel, wxID_ANY, "Redo");
    modifySizer->Add(addButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(deleteButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(connectButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(disconnectButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(contractButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(subdivideButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->AddStretchSpacer(1);
    modifySizer->Add(undoButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(redoButton, 0, wxALIGN_CENTER | wxLEFT | wxRIGHT | wxTOP | wxBOTTOM, 5);
    modifyPanel->SetSizerAndFit(modifySizer);
    notebook->AddPage(modifyPanel, "Edit");

    auto drawPanel = new wxPanel(notebook);
    auto drawSizer = new wxBoxSizer(wxHORIZONTAL);
    auto anchorButton = new wxButton(drawPanel, wxID_ANY, "Anchor");
    auto freeButton = new wxButton(drawPanel, wxID_ANY, "Free");
    auto autoVertexPositioningCheckbox = new wxCheckBox(drawPanel, wxID_ANY, "Automatic vertex positioning");
    auto showFPSCheckbox = new wxCheckBox(drawPanel, wxID_ANY, "Show FPS");
    drawSizer->Add(anchorButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(freeButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(autoVertexPositioningCheckbox, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->AddStretchSpacer(1);
    drawSizer->Add(showFPSCheckbox, 0, wxALIGN_CENTER | wxRIGHT | wxTOP | wxBOTTOM, 5);
    drawPanel->SetSizerAndFit(drawSizer);
    notebook->AddPage(drawPanel, "View");

    auto testPanel = new wxPanel(notebook);
    auto testSizer = new wxBoxSizer(wxHORIZONTAL);
    auto initButton = new wxButton(testPanel, wxID_ANY, "Create");
    auto selectButton = new wxButton(testPanel, wxID_ANY, "Select");
    testSizer->Add(initButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    testSizer->Add(selectButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    testSizer->AddStretchSpacer(1);
    testPanel->SetSizerAndFit(testSizer);
    notebook->AddPage(testPanel, "Test");

    sizer->Add(nameLabel, 0, wxALIGN_CENTER | wxTOP | wxBOTTOM, 5);
    sizer->Add(notebook, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    this->SetSizerAndFit(sizer);

    openButton->Bind(wxEVT_BUTTON, &GraphPanel::OpenFromFile, this);
    saveButton->Bind(wxEVT_BUTTON, &GraphPanel::SaveToFile, this);
    initButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        auto graph = utils::GraphFactory::random_graph(5, 0.5f);

        manager.Initialize(std::move(graph));

        auto& vertexPositions2D = manager.Positions2D();
        canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);
        auto& vertexStates = manager.States();
		canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
        auto edges = manager.GetEdges();
		canvas->SetEdges(edges.data(), edges.size() / 2);
    });
    selectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.HandleClick();
        auto& vertexStates = manager.States();
        canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
		canvas->Refresh();
    });

    lastFrameTime = animationClock::now();
    Bind(wxEVT_IDLE, &GraphPanel::OnIdle, this);
}

const core::Graph& GraphPanel::GetGraph() const {
    return manager.Graph();
}

void GraphPanel::OnMatchingStart() {
    openButton->Disable();
    addButton->Disable();
    deleteButton->Disable();
    connectButton->Disable();
    disconnectButton->Disable();
    contractButton->Disable();
    subdivideButton->Disable();
    undoButton->Disable();
    redoButton->Disable();
}

void GraphPanel::OnMatchingEnd() {
    openButton->Enable();
    addButton->Enable();
    deleteButton->Enable();
    connectButton->Enable();
    disconnectButton->Enable();
    contractButton->Enable();
    subdivideButton->Enable();
    undoButton->Enable();
    redoButton->Enable();
}

void GraphPanel::OpenFromFile(wxCommandEvent& event) {
    auto fileDialog = new wxFileDialog(this, "Choose a file to open", wxEmptyString, wxEmptyString,
                                       "Graph6 files (*.g6)|*.g6", wxFD_OPEN);

    if (fileDialog->ShowModal() == wxID_OK) {
        wxFileInputStream inputStream(fileDialog->GetPath());

        if (!inputStream.IsOk()) {
            wxMessageBox("Could not open file: " + fileDialog->GetFilename());
        } else {
            wxTextInputStream graph6Stream(inputStream, wxT("\x09"), wxConvUTF8);
            try {
                auto graph = core::Graph6Serializer::Deserialize(graph6Stream.ReadLine().ToStdString());
                fileInfoLabel->SetLabel(fileDialog->GetFilename() + " (Graph6)");

				manager.Initialize(std::move(graph));

				auto& vertexPositions2D = manager.Positions2D();
				canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);
				auto& vertexStates = manager.States();
				canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
				auto edges = manager.GetEdges();
				canvas->SetEdges(edges.data(), edges.size() / 2);
                fileOpenCallback();
            } catch (const core::graph6FormatError& err) {
                wxMessageBox("Could not open file " + fileDialog->GetFilename() + "\nError: " + err.what());
            }
        }
    }

    fileDialog->Destroy();
}

void GraphPanel::SaveToFile(wxCommandEvent& event) {
    auto fileDialog = new wxFileDialog(this, "Save the graph to file", wxEmptyString, wxEmptyString,
                                       "Graph6 files (*.g6)|*.g6", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);

    if (fileDialog->ShowModal() == wxID_OK) {
        wxFileOutputStream outputStream(fileDialog->GetPath());

        if (!outputStream.IsOk()) {
            wxMessageBox("Cannot save the graph in file: " + fileDialog->GetFilename());
        } else {
            wxTextOutputStream graph6Stream(outputStream, wxEOL_NATIVE, wxConvUTF8);
            try {
                graph6Stream << core::Graph6Serializer::Serialize(manager.Graph());
            } catch (const core::graph6FormatError& err) {
                wxMessageBox("Could save to file " + fileDialog->GetFilename() + "\nError: " + err.what());
            }
        }
    }

    fileDialog->Destroy();
}

void GraphPanel::OnIdle(wxIdleEvent& event) {
    auto now = animationClock::now();
    std::chrono::duration<float> elapsedSeconds = now - lastFrameTime;
    lastFrameTime = now;
    if (elapsedSeconds.count() > 0.5f) {
        event.RequestMore();
        return;
    }

    manager.UpdatePositions(elapsedSeconds.count());

    auto& vertexPositions2D = manager.Positions2D();
    canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);

    canvas->Refresh();
    event.RequestMore();
}
