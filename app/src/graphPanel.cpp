#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "wx/notebook.h"
#include "wx/numformatter.h"
#include "utils.h"
#include <numeric>

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> fileOpenCallback)
    : wxPanel(parent), fileOpenCallback(fileOpenCallback) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        canvas = new GraphCanvas(this, vAttrs);
        canvas->SetInitialSize(wxSize(0, 360));
        sizer->Add(canvas, 1, wxEXPAND);
        canvas->Bind(wxEVT_LEFT_DOWN, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_LEFT_DCLICK, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_MOTION, &GraphPanel::OnCanvasMotion, this);
        canvas->Bind(wxEVT_LEFT_UP, [this](wxMouseEvent& event) { manager.OnDrop(); });
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
    deleteButton = new wxButton(modifyPanel, wxID_ANY, "Delete");
    connectButton = new wxButton(modifyPanel, wxID_ANY, "Connect");
    disconnectButton = new wxButton(modifyPanel, wxID_ANY, "Disconnect");
    contractButton = new wxButton(modifyPanel, wxID_ANY, "Contract");
    subdivideButton = new wxButton(modifyPanel, wxID_ANY, "Subdivide");
    undoButton = new wxButton(modifyPanel, wxID_ANY, "Undo");
    redoButton = new wxButton(modifyPanel, wxID_ANY, "Redo");
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

    connectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.ConnectSelection();

        auto edges = manager.GetEdges();
        canvas->SetEdges(edges.data(), edges.size() / 2);
    });
    disconnectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.DisconnectSelection();

        auto edges = manager.GetEdges();
        canvas->SetEdges(edges.data(), edges.size() / 2);
    });

    auto drawPanel = new wxPanel(notebook);
    auto drawSizer = new wxBoxSizer(wxHORIZONTAL);
    auto anchorButton = new wxButton(drawPanel, wxID_ANY, "Anchor");
    auto freeButton = new wxButton(drawPanel, wxID_ANY, "Free");
    autoVertexPositioningCheckbox = new wxCheckBox(drawPanel, wxID_ANY, "Automatic vertex positioning");
    FPSInfoLabel = new wxStaticText(drawPanel, wxID_ANY, "FPS: 00000");
    drawSizer->Add(anchorButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(freeButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(autoVertexPositioningCheckbox, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->AddStretchSpacer(1);
    drawSizer->Add(FPSInfoLabel, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawPanel->SetSizerAndFit(drawSizer);
    notebook->AddPage(drawPanel, "View");

    anchorButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.AnchorSelection();

        auto& vertexStates = manager.States();
        canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
        canvas->Refresh();
    });
    freeButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.FreeSelection();

        auto& vertexStates = manager.States();
        canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
        canvas->Refresh();
    });
    autoVertexPositioningCheckbox->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& event) { manager.Stop(); });

    auto testPanel = new wxPanel(notebook);
    auto testSizer = new wxBoxSizer(wxHORIZONTAL);
    auto initButton = new wxButton(testPanel, wxID_ANY, "Create");
    testSizer->Add(initButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
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
        auto [boundingWidth, boundingHeight] = manager.BoundingSize();
        auto [centerX, centerY] = manager.Center();
        auto& vertexStates = manager.States();
        auto edges = manager.GetEdges();

        canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);
        canvas->SetBoundingSize(boundingWidth, boundingHeight);
        canvas->SetCenterPosition(centerX, centerY);
        canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
        canvas->SetEdges(edges.data(), edges.size() / 2);
    });

    lastFrameTime = animationClock::now();
    Bind(wxEVT_IDLE, &GraphPanel::OnIdle, this);
}

const core::Graph& GraphPanel::GetGraph() const {
    return manager.Graph();
}

void GraphPanel::OnMatchingStart() {
    openButton->Disable();
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
                auto [boundingWidth, boundingHeight] = manager.BoundingSize();
                auto [centerX, centerY] = manager.Center();
                auto& vertexStates = manager.States();
                auto edges = manager.GetEdges();

                canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);
                canvas->SetBoundingSize(boundingWidth, boundingHeight);
                canvas->SetCenterPosition(centerX, centerY);
                canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
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

    if (autoVertexPositioningCheckbox->IsChecked()) manager.UpdatePositions(elapsedSeconds.count());

    fpsArray[fpsIndex++] = (int)(1.0 / elapsedSeconds.count());
    if (fpsIndex >= FPS_ANALYSIS_COUNT) fpsIndex = 0;
    int avgFPS = (int)(std::accumulate(fpsArray, fpsArray + FPS_ANALYSIS_COUNT, 0) / FPS_ANALYSIS_COUNT);
    FPSInfoLabel->SetLabel(wxString::Format("FPS: %5d", avgFPS));

    auto& vertexPositions2D = manager.Positions2D();
    auto [boundingWidth, boundingHeight] = manager.BoundingSize();
    auto [centerX, centerY] = manager.Center();

    canvas->SetVertexPositions(vertexPositions2D.data(), vertexPositions2D.size() / 2);
    canvas->SetBoundingSize(boundingWidth, boundingHeight);
    canvas->SetCenterPosition(centerX, centerY);

    canvas->Refresh();
    event.RequestMore();
}

void GraphPanel::OnCanvasClick(wxMouseEvent& event) {

    auto point = event.GetPosition();
    auto [boundingWidth, boundingHeight] = manager.BoundingSize();
    auto [centerX, centerY] = manager.Center();
    auto [canvasWidth, canvasHeight] = canvas->CanvasSize();

    float ratio = std::max(boundingWidth / (canvasWidth - 3 * canvas->NODE_RADIUS),
                           boundingHeight / (canvasHeight - 3 * canvas->NODE_RADIUS));
    float worldX = (point.x - canvasWidth / 2) * ratio + centerX;
    float worldY = (canvasHeight / 2 - point.y) * ratio + centerY;
    manager.HandleClick(worldX, worldY, canvas->NODE_RADIUS * 0.5f * ratio, event.ControlDown(), event.ButtonDClick());

    auto& vertexStates = manager.States();
    canvas->SetVertexStates(vertexStates.data(), vertexStates.size());
}

void GraphPanel::OnCanvasMotion(wxMouseEvent& event) {
    if (!event.Dragging() || !event.LeftIsDown()) {
        prevMousePoint = std::nullopt;
        return;
    }

    auto mousePoint = event.GetPosition();
    if (prevMousePoint.has_value()) {
        auto [boundingWidth, boundingHeight] = manager.BoundingSize();
        auto [canvasWidth, canvasHeight] = canvas->CanvasSize();

        float ratio = std::max(boundingWidth / (canvasWidth - 3 * canvas->NODE_RADIUS),
                               boundingHeight / (canvasHeight - 3 * canvas->NODE_RADIUS));
        manager.OnDrag((mousePoint.x - prevMousePoint.value().x) * ratio,
                       -(mousePoint.y - prevMousePoint.value().y) * ratio);
    }
    prevMousePoint = mousePoint;
}
