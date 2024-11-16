#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include <chrono>
#include "utils.h"

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(wxWindow* parent, const wxString& title) : wxPanel(parent), graph(0) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        canvas = new GraphCanvas(this, vAttrs);
        canvas->SetInitialSize(wxSize(0, 360));
        sizer->Add(canvas, 1, wxEXPAND);
    }

    auto nameLabel = new wxStaticText(this, wxID_ANY, title);

    auto fileBoxSizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "Files");
    auto saveButton = new wxButton(fileBoxSizer->GetStaticBox(), wxID_ANY, "Save");
    auto openButton = new wxButton(fileBoxSizer->GetStaticBox(), wxID_ANY, "Open");
    auto fileInfoLabel = new wxStaticText(fileBoxSizer->GetStaticBox(), wxID_ANY, "Open a file to load the graph.");
    auto vertexCountInput = new wxTextCtrl(fileBoxSizer->GetStaticBox(), wxID_ANY);
    vertexCountInput->SetHint("Vertex count");
    vertexCountInput->SetMinSize(wxSize(120, wxDefaultCoord));
    auto loadButton = new wxButton(fileBoxSizer->GetStaticBox(), wxID_ANY, "Load");
    fileBoxSizer->Add(saveButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    fileBoxSizer->Add(openButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    fileBoxSizer->Add(fileInfoLabel, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    fileBoxSizer->AddStretchSpacer(1);
    fileBoxSizer->Add(vertexCountInput, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    fileBoxSizer->Add(loadButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM | wxRIGHT, 5);

    auto modifyBoxSizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "Modifications");
    auto addButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Add");
    auto deleteButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Delete");
    auto connectButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Connect");
    auto disconnectButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Disconnect");
    auto contractButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Contract");
    auto subdivideButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Subdivide");
    auto undoButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Undo");
    auto redoButton = new wxButton(modifyBoxSizer->GetStaticBox(), wxID_ANY, "Redo");
    modifyBoxSizer->Add(addButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(deleteButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(connectButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(disconnectButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(contractButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(subdivideButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->AddStretchSpacer(1);
    modifyBoxSizer->Add(undoButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    modifyBoxSizer->Add(redoButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM | wxRIGHT, 5);

    auto drawBoxSizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "Drawing");
    auto anchorButton = new wxButton(drawBoxSizer->GetStaticBox(), wxID_ANY, "Anchor");
    auto freeButton = new wxButton(drawBoxSizer->GetStaticBox(), wxID_ANY, "Free");
    auto autoVertexPositioningCheckbox = 
        new wxCheckBox(drawBoxSizer->GetStaticBox(), wxID_ANY, "Automatic vertex positioning");
    auto showFPSCheckbox = new wxCheckBox(drawBoxSizer->GetStaticBox(), wxID_ANY, "Show FPS");
    drawBoxSizer->Add(anchorButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    drawBoxSizer->Add(freeButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    drawBoxSizer->Add(autoVertexPositioningCheckbox, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    drawBoxSizer->AddStretchSpacer(1);
    drawBoxSizer->Add(showFPSCheckbox, 0, wxALIGN_CENTER | wxRIGHT | wxBOTTOM, 5);

    auto testBoxSizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "Testing");
    auto initButton = new wxButton(testBoxSizer->GetStaticBox(), wxID_ANY, "Create");
    auto colorButton = new wxButton(testBoxSizer->GetStaticBox(), wxID_ANY, "Color");
    testBoxSizer->Add(initButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    testBoxSizer->Add(colorButton, 0, wxALIGN_CENTER | wxLEFT | wxBOTTOM, 5);
    testBoxSizer->AddStretchSpacer(1);

    sizer->Add(nameLabel, 0, wxALIGN_CENTER | wxTOP, 5);
    sizer->Add(fileBoxSizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
    sizer->Add(modifyBoxSizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
    sizer->Add(drawBoxSizer, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    sizer->Add(testBoxSizer, 0, wxEXPAND | wxLEFT | wxRIGHT, 5);
    this->SetSizerAndFit(sizer);

    openButton->Bind(wxEVT_BUTTON, [this, fileInfoLabel](wxCommandEvent& event) { 
        auto fileDialog = new wxFileDialog(
            this, "Choose a file to open", wxEmptyString, wxEmptyString,
            "Graph6 files (*.g6)|*.g6", wxFD_OPEN);

        if (fileDialog->ShowModal() == wxID_OK) {
            wxFileInputStream inputStream(fileDialog->GetPath());
            
            if (!inputStream.IsOk()) {
                wxMessageBox("Could not open file: " + fileDialog->GetFilename());
            } else {
                wxTextInputStream graph6Stream(inputStream, wxT("\x09"), wxConvUTF8);
                try {
                    graph = core::Graph6Serializer::Deserialize(graph6Stream.ReadLine().ToStdString());
                    fileInfoLabel->SetLabel(fileDialog->GetFilename() + " (Graph6)");
                    InitGraphSimulation();
                } catch (const core::graph6FormatError& err) {
                    wxMessageBox(err.what());
                }
            }
        }

        fileDialog->Destroy();
    });

    saveButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        auto fileDialog = new wxFileDialog(
            this, "Save the graph to file", wxEmptyString, wxEmptyString,
            "Graph6 files (*.g6)|*.g6", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);

        if (fileDialog->ShowModal() == wxID_OK) {
            wxFileOutputStream outputStream(fileDialog->GetPath());
            
            if (!outputStream.IsOk()) {
                wxMessageBox("Cannot save the graph in file: " + fileDialog->GetFilename());
            } else {
                wxTextOutputStream graph6Stream(outputStream, wxEOL_NATIVE, wxConvUTF8);
                try {
                    graph6Stream << core::Graph6Serializer::Serialize(graph);
                } catch (const core::graph6FormatError& err) {
                    wxMessageBox(err.what());
                }
            }
        }

        fileDialog->Destroy();
    });

    initButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) { 
        graph = utils::GraphFactory::random_graph(5, 0.5f);
        InitGraphSimulation();
    });

    colorButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        wxColourData colorData;
        colorData.SetColour(this->canvas->vertexColor);
        wxColourDialog dialog(this, &colorData);

        if (dialog.ShowModal() == wxID_OK) {
            canvas->vertexColor = dialog.GetColourData().GetColour();
            canvas->Refresh();
        }
    });

    lastFrameTime = animationClock::now();
    Bind(wxEVT_IDLE, &GraphPanel::OnIdle, this);
}

GraphPanel::~GraphPanel() {
    if (vertexPositions2D[0])  delete[] vertexPositions2D[0];
    if (vertexPositions2D[1])  delete[] vertexPositions2D[1];
    if (vertexVelocities2D[0]) delete[] vertexVelocities2D[0];
    if (vertexVelocities2D[1]) delete[] vertexVelocities2D[1];
}

const core::Graph& GraphPanel::GetGraph() const {
    return graph;
}

void GraphPanel::InitGraphSimulation() {
    if (vertexPositions2D[0])  delete[] vertexPositions2D[0];
    if (vertexPositions2D[1])  delete[] vertexPositions2D[1];
    if (vertexVelocities2D[0]) delete[] vertexVelocities2D[0];
    if (vertexVelocities2D[1]) delete[] vertexVelocities2D[1];

    vertexPositions2D[0] = new float[2 * graph.size()];
    vertexPositions2D[1] = new float[2 * graph.size()];
    vertexVelocities2D[0] = new float[2 * graph.size()];
    vertexVelocities2D[1] = new float[2 * graph.size()];

    std::vector<unsigned int> edges;
    for (unsigned int i = 0; i < graph.size(); i++) {
        for (unsigned int j = 0; j < i; j++) {
            if (graph.has_edge(i, j)) {
                edges.push_back(i);
                edges.push_back(j);
            }
        }
        vertexPositions2D[readBufferId][2 * i] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexPositions2D[readBufferId][2 * i + 1] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
    }
    canvas->SetVertexPositions(vertexPositions2D[readBufferId], graph.size());
    canvas->SetEdges(edges.data(), edges.size() / 2);
}

void GraphPanel::OnIdle(wxIdleEvent& event) {
    auto now = animationClock::now();
    std::chrono::duration<float> elapsedSeconds = now - lastFrameTime;
    lastFrameTime = now;
    if (elapsedSeconds.count() > 0.5f) {
        event.RequestMore();
        return;
    }

    for (unsigned int i = 0; i < graph.size(); i++) {
        float x = vertexPositions2D[readBufferId][2 * i];
        float y = vertexPositions2D[readBufferId][2 * i + 1];

        float distOrigin = sqrtf(x * x + y * y);
        float gravityCoefficient = distOrigin < 0.1f ? 0.0f : C[3] / (distOrigin * distOrigin * distOrigin);

        float a_x = gravityCoefficient * x;
        float a_y = gravityCoefficient * y;

        for (unsigned int j = 0; j < graph.size(); j++) {
            if (i == j) continue;
            float dx = x - vertexPositions2D[readBufferId][2 * j];
            float dy = y - vertexPositions2D[readBufferId][2 * j + 1];
            float dist = sqrtf(dx * dx + dy * dy);
            
            float springCoefficient = !graph.has_edge(i, j) || !graph.has_edge(j, i) || dist < 0.01f
                ? 0.0f : C[0] * logf(dist / C[1]) / dist;
            float repelCoefficient = dist < 0.0001f ? 0.0f : C[2] / (dist * dist * dist);

            a_x += (repelCoefficient + springCoefficient) * x;
            a_y += (repelCoefficient + springCoefficient) * y;
        }

        float v_x = vertexVelocities2D[readBufferId][2 * i];
        float v_y = vertexVelocities2D[readBufferId][2 * i + 1];

		float tractionCoefficient = C[4];
		a_x += tractionCoefficient * v_x;
		a_y += tractionCoefficient * v_y;
        
        vertexVelocities2D[1 - readBufferId][2 * i] = v_x + a_x * elapsedSeconds.count();
        vertexVelocities2D[1 - readBufferId][2 * i + 1] = v_y + a_y * elapsedSeconds.count();
        vertexPositions2D[1 - readBufferId][2 * i] = x + v_x * elapsedSeconds.count();
        vertexPositions2D[1 - readBufferId][2 * i + 1] = y + v_y * elapsedSeconds.count();
    }

    readBufferId = 1 - readBufferId;
    canvas->SetVertexPositions(vertexPositions2D[readBufferId], graph.size());

    canvas->Refresh();
    event.RequestMore();
}
