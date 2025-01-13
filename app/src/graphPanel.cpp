#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "wx/notebook.h"
#include "wx/numformatter.h"
#include <numeric>
#include <filesystem>
#include "image.h"

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(
    wxWindow* parent, const wxString& title, std::function<void()> clearMatchingCallback,
    std::function<std::vector<std::optional<std::pair<float, float>>>()> getMatchingAlignmentCallback)
    : wxPanel(parent), clearMatchingCallback(clearMatchingCallback), canModifyGraph(true) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    auto nameLabel = new wxStaticText(this, wxID_ANY, title);

    auto notebook = new wxNotebook(this, wxID_ANY);

    auto filePanel = new wxPanel(notebook);
    auto fileSizer = new wxBoxSizer(wxHORIZONTAL);
    auto saveButton = new wxButton(filePanel, wxID_ANY, "Save");
    openButton = new wxButton(filePanel, wxID_ANY, "Open");
    fileInfoLabel = new wxStaticText(filePanel, wxID_ANY, "Open a file to load the graph.");
    vertexCountInput = new wxTextCtrl(filePanel, wxID_ANY);
    vertexCountInput->SetHint("Vertex count");
    vertexCountInput->SetMinSize(wxSize(120, wxDefaultCoord));
    vertexCountInput->Disable();
    loadButton = new wxButton(filePanel, wxID_ANY, "Load");
    loadButton->Disable();
    fileSizer->Add(saveButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(openButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(fileInfoLabel, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->AddStretchSpacer(1);
    fileSizer->Add(vertexCountInput, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    fileSizer->Add(loadButton, 0, wxALIGN_CENTER | wxLEFT | wxRIGHT | wxTOP | wxBOTTOM, 5);
    filePanel->SetSizerAndFit(fileSizer);
    notebook->AddPage(filePanel, "File");

    loadButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        auto vertexCount = 0;
        auto vertexCountString = vertexCountInput->GetValue();
        if (!vertexCountString.ToInt(&vertexCount)) {
            wxMessageBox("Could not read requested vertex count.\nError: '" + vertexCountString + "' is not a number.");
            return;
        }

        try {
            auto [graph, vertexPositions] = image::grapherize(pathToImage, vertexCount);
            manager.Initialize(std::move(graph), std::move(vertexPositions));
        } catch (const std::runtime_error& err) {
            wxMessageBox("Could not load graph from given image.");
            wxLogDebug("ERROR: %s", err.what());
            return;
        }
        OnGraphUpdate();
    });

    auto modifyPanel = new wxPanel(notebook);
    auto modifySizer = new wxBoxSizer(wxHORIZONTAL);
    deleteButton = new wxButton(modifyPanel, wxID_ANY, "Delete");
    connectButton = new wxButton(modifyPanel, wxID_ANY, "Connect");
    disconnectButton = new wxButton(modifyPanel, wxID_ANY, "Disconnect");
    contractButton = new wxButton(modifyPanel, wxID_ANY, "Contract");
    subdivideButton = new wxButton(modifyPanel, wxID_ANY, "Subdivide");
    modifySizer->Add(deleteButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(connectButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(disconnectButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(contractButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->Add(subdivideButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    modifySizer->AddStretchSpacer(1);
    modifyPanel->SetSizerAndFit(modifySizer);
    notebook->AddPage(modifyPanel, "Edit");

    deleteButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.DeleteSelection();
        OnGraphUpdate();
    });
    connectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.ConnectSelection();
        OnGraphUpdate();
    });
    disconnectButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.DisconnectSelection();
        OnGraphUpdate();
    });
    contractButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.ContractSelection();
        if (manager.IsAnimationRunning()) {
            this->clearMatchingCallback();
            DisableGraphModifications();
        } else {
            OnGraphUpdate();
        }
    });
    subdivideButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.SubdivideSelection();
        OnGraphUpdate();
    });

    auto drawPanel = new wxPanel(notebook);
    auto drawSizer = new wxBoxSizer(wxHORIZONTAL);
    auto anchorButton = new wxButton(drawPanel, wxID_ANY, "Anchor");
    auto freeButton = new wxButton(drawPanel, wxID_ANY, "Free");
    autoVertexPositioningCheckbox = new wxCheckBox(drawPanel, wxID_ANY, "Automatic vertex positioning");
    alignButton = new wxButton(drawPanel, wxID_ANY, "Align");
    alignButton->Disable();
    FPSInfoLabel = new wxStaticText(drawPanel, wxID_ANY, "FPS: 00000");
    drawSizer->Add(anchorButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(freeButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(autoVertexPositioningCheckbox, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->AddStretchSpacer(1);
    drawSizer->Add(alignButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    drawSizer->Add(FPSInfoLabel, 0, wxALIGN_CENTER | wxALL, 5);
    drawPanel->SetSizerAndFit(drawSizer);
    notebook->AddPage(drawPanel, "View");

    anchorButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.AnchorSelection();
        canvas->SetVertexStates(manager.GetStates().data());
    });
    freeButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        manager.FreeSelection();
        canvas->SetVertexStates(manager.GetStates().data());
    });
    autoVertexPositioningCheckbox->Bind(wxEVT_CHECKBOX, [this](wxCommandEvent& event) { manager.Stop(); });
    alignButton->Bind(wxEVT_BUTTON, [this, getMatchingAlignmentCallback](wxCommandEvent& event) {
        auto alignment = getMatchingAlignmentCallback();
        manager.AlignNodes(alignment);
        canvas->SetVertexStates(manager.GetStates().data());
    });

    sizer->Add(nameLabel, 0, wxALIGN_CENTER | wxTOP | wxBOTTOM, 5);
    sizer->Add(notebook, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        canvas = new GraphCanvas(this, vAttrs);
        canvas->SetInitialSize(wxSize(0, 360));
        sizer->Add(canvas, 1, wxEXPAND);

        canvas->Bind(wxEVT_LEFT_DOWN, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_LEFT_UP, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_LEFT_DCLICK, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_RIGHT_DOWN, &GraphPanel::OnCanvasClick, this);
        canvas->Bind(wxEVT_RIGHT_UP, &GraphPanel::OnCanvasClick, this);

        canvas->Bind(wxEVT_MOTION, &GraphPanel::OnCanvasMotion, this);
    }

    this->SetSizerAndFit(sizer);

    openButton->Bind(wxEVT_BUTTON, &GraphPanel::OpenFromFile, this);
    saveButton->Bind(wxEVT_BUTTON, &GraphPanel::SaveToFile, this);

    lastFrameTime = animationClock::now();
    Bind(wxEVT_IDLE, &GraphPanel::OnIdle, this);
}

const GraphManager& GraphPanel::Manager() const {
    return manager;
}

void GraphPanel::OnGraphUpdate() {
    canvas->SetVertexCount(manager.RenderedVertexCount());
    auto& vertexPositions2D = manager.Positions2D();
    canvas->SetVertexPositions(vertexPositions2D.data());
    canvas->SetVertexStates(manager.GetStates().data());
    auto edges = manager.GetEdges();
    canvas->SetEdges(edges.data(), edges.size() / 2);

    this->clearMatchingCallback();
}

void GraphPanel::EnableGraphModifications() {
    canModifyGraph = true;
    openButton->Enable();
    if (pathToImage != "") loadButton->Enable();
    deleteButton->Enable();
    connectButton->Enable();
    disconnectButton->Enable();
    contractButton->Enable();
    subdivideButton->Enable();
}

void GraphPanel::DisableGraphModifications() {
    canModifyGraph = false;
    openButton->Disable();
    loadButton->Disable();
    deleteButton->Disable();
    connectButton->Disable();
    disconnectButton->Disable();
    contractButton->Disable();
    subdivideButton->Disable();
}

void GraphPanel::OnMatchingStart() {
    DisableGraphModifications();
}

void GraphPanel::OnMatchingEnd() {
    EnableGraphModifications();
    alignButton->Disable();

    std::vector<unsigned int> labelling(manager.Graph().size(), 0);
    auto renderedLabelling = manager.GetRenderedLabelling(labelling);
    canvas->SetVertexLabels(renderedLabelling.data());
}

void GraphPanel::OnMatchingEnd(const std::vector<unsigned int>& labelling) {
    EnableGraphModifications();
    alignButton->Enable();

    auto renderedLabelling = manager.GetRenderedLabelling(labelling);
    canvas->SetVertexLabels(renderedLabelling.data());
}

void GraphPanel::OpenFromFile(wxCommandEvent& event) {
    auto fileDialog =
        new wxFileDialog(this, "Choose a file to open", wxEmptyString, wxEmptyString,
                         "Graph6 files (*.g6)|*.g6|Image files (*.png; *.jpg; *.jpeg)|*.png;*.jpg;*.jpeg", wxFD_OPEN);

    if (fileDialog->ShowModal() != wxID_OK) {
        fileDialog->Destroy();
        return;
    }

    if (std::filesystem::path((const char*)fileDialog->GetPath()).extension() == ".g6") {
        wxFileInputStream inputStream(fileDialog->GetPath());

        if (!inputStream.IsOk()) {
            wxMessageBox("Could not open file: " + fileDialog->GetFilename());
            fileDialog->Destroy();
            return;
        }

        wxTextInputStream graph6Stream(inputStream, wxT("\x09"), wxConvUTF8);

        try {
            auto graph = core::Graph6Serializer::Deserialize(graph6Stream.ReadLine().ToStdString());
            fileInfoLabel->SetLabel(fileDialog->GetFilename() + " (Graph6)");
            manager.Initialize(std::move(graph));
        } catch (const core::graph6FormatError& err) {
            wxMessageBox("Could not open file " + fileDialog->GetFilename() + "\nError: " + err.what());
            fileDialog->Destroy();
            return;
        }

        OnGraphUpdate();

        pathToImage = "";
        vertexCountInput->Disable();
        loadButton->Disable();
    } else {
        fileInfoLabel->SetLabel(fileDialog->GetFilename() + " (Image)");

        pathToImage = fileDialog->GetPath();
        vertexCountInput->Enable();
        if (canModifyGraph) loadButton->Enable();
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

    if (autoVertexPositioningCheckbox->IsChecked()) {
        manager.UpdatePositions(elapsedSeconds.count(), vertexDragging);
    }

    if (manager.UpdateRenderedPositions(elapsedSeconds.count())) {
        canvas->SetVertexCount(manager.RenderedVertexCount());
        canvas->SetVertexStates(manager.GetStates().data());
        auto edges = manager.GetEdges();
        canvas->SetEdges(edges.data(), edges.size() / 2);
        EnableGraphModifications();
    }

    if (!vertexDragging) manager.UpdateBounds(elapsedSeconds.count());

    auto& vertexPositions2D = manager.Positions2D();
    auto [boundingWidth, boundingHeight] = manager.BoundingSize();
    auto [centerX, centerY] = manager.Center();

    canvas->SetVertexPositions(vertexPositions2D.data());
    canvas->SetBoundingSize(boundingWidth, boundingHeight);
    canvas->SetCenterPosition(centerX, centerY);

    fpsArray[fpsIndex++] = (int)(1.0 / elapsedSeconds.count());
    if (fpsIndex >= FPS_ANALYSIS_COUNT) fpsIndex = 0;
    int avgFPS = (int)(std::accumulate(fpsArray, fpsArray + FPS_ANALYSIS_COUNT, 0) / FPS_ANALYSIS_COUNT);
    FPSInfoLabel->SetLabel(wxString::Format("FPS: %5d", avgFPS));

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
    float worldNodeRadius = canvas->NODE_RADIUS * 0.5f * ratio;

    if (event.LeftDown()) {

        if (!event.ControlDown()) manager.ClearSelection();
        auto collidingNodes = manager.GetCollidingNodes(worldX, worldY, worldNodeRadius);
        if (!event.ControlDown()) manager.SelectNodes(collidingNodes);

        if (!event.ControlDown()) canvas->SetVertexStates(manager.GetStates().data());

        areaSelectionStartPoint = collidingNodes.empty() ? std::make_optional(point) : std::nullopt;
        vertexDragging = !collidingNodes.empty();

    } else if (event.LeftUp()) {

        if (areaSelectionStartPoint.has_value()) {
            float worldStartX = (areaSelectionStartPoint.value().x - canvasWidth / 2) * ratio + centerX;
            float worldStartY = (canvasHeight / 2 - areaSelectionStartPoint.value().y) * ratio + centerY;
            auto collidingNodes = manager.GetCollidingNodes(worldStartX, worldStartY, worldX, worldY);
            manager.SelectNodes(collidingNodes);
        } else if (event.ControlDown()) {
            auto collidingNodes = manager.GetCollidingNodes(worldX, worldY, worldNodeRadius);
            auto last = std::unique(collidingNodes.begin(), collidingNodes.end());
            collidingNodes.erase(last, collidingNodes.end());
            manager.ToggleNodes(collidingNodes);
        }

        canvas->SetVertexStates(manager.GetStates().data());
        canvas->ClearUtilityLoop();

        areaSelectionStartPoint = std::nullopt;
        vertexDragging = false;

    } else if (event.LeftDClick() && canModifyGraph) {

        manager.AddVertex(worldX, worldY);

        OnGraphUpdate();

    } else if (event.RightDown() && canModifyGraph) {

        auto collidingNodes = manager.GetCollidingNodes(worldX, worldY, worldNodeRadius);
        if (!collidingNodes.empty()) connectionStartVertex = collidingNodes.front();

    } else if (event.RightUp() && connectionStartVertex.has_value()) {

        auto collidingNodes = manager.GetCollidingNodes(worldX, worldY, worldNodeRadius);
        if (!collidingNodes.empty()) {
            manager.ConnectNodes(connectionStartVertex.value(), collidingNodes.front());

            OnGraphUpdate();
        }
        canvas->ClearUtilityLoop();

        connectionStartVertex = std::nullopt;
    }
}

void GraphPanel::OnCanvasMotion(wxMouseEvent& event) {
    if (!event.Dragging()) {
        prevMousePoint = std::nullopt;
        return;
    }

    auto mousePoint = event.GetPosition();

    if (prevMousePoint.has_value()) {
        auto [boundingWidth, boundingHeight] = manager.BoundingSize();
        auto [canvasWidth, canvasHeight] = canvas->CanvasSize();
        float ratio = std::max(boundingWidth / (canvasWidth - 3 * canvas->NODE_RADIUS),
                               boundingHeight / (canvasHeight - 3 * canvas->NODE_RADIUS));

        if (event.LeftIsDown()) {
            if (areaSelectionStartPoint.has_value()) {

                float areaSelectionPositions2D[8] = {
                    (float)areaSelectionStartPoint.value().x,
                    (float)areaSelectionStartPoint.value().y,
                    (float)mousePoint.x,
                    (float)areaSelectionStartPoint.value().y,
                    (float)mousePoint.x,
                    (float)mousePoint.y,
                    (float)areaSelectionStartPoint.value().x,
                    (float)mousePoint.y,
                };
                canvas->SetUtilityLoop(areaSelectionPositions2D, 4);

            } else {

                float worldDX = (mousePoint.x - prevMousePoint.value().x) * ratio;
                float worldDY = -(mousePoint.y - prevMousePoint.value().y) * ratio;

                manager.OnVertexDrag(worldDX, worldDY);
            }
        } else if (event.RightIsDown() && connectionStartVertex.has_value()) {

            auto [centerX, centerY] = manager.Center();
            auto [worldStartX, worldStartY] = manager.Position2D(connectionStartVertex.value());
            auto startX = (worldStartX - centerX) / ratio + canvasWidth / 2;
            auto endX = (centerY - worldStartY) / ratio + canvasHeight / 2;
            float edgeConnectionEndsPositions2D[4] = {startX, endX, (float)mousePoint.x, (float)mousePoint.y};
            canvas->SetUtilityLoop(edgeConnectionEndsPositions2D, 2);
        }
    }
    prevMousePoint = mousePoint;
}

void GraphPanel::UpdateDrawingSettings(GraphDrawingSettings settings) {
    manager.UpdateSettings(settings);
}
