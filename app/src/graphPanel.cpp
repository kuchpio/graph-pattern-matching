#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "wx/notebook.h"
#include "utils.h"

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(wxWindow* parent, const wxString& title, std::function<void()> fileOpenCallback) 
    : wxPanel(parent), graph(0), fileOpenCallback(fileOpenCallback) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

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
    auto colorButton = new wxButton(testPanel, wxID_ANY, "Color");
    testSizer->Add(initButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    testSizer->Add(colorButton, 0, wxALIGN_CENTER | wxLEFT | wxTOP | wxBOTTOM, 5);
    testSizer->AddStretchSpacer(1);
    testPanel->SetSizerAndFit(testSizer);
    notebook->AddPage(testPanel, "Test");

    sizer->Add(nameLabel, 0, wxALIGN_CENTER | wxTOP | wxBOTTOM, 5);
    sizer->Add(notebook, 0, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, 5);
    this->SetSizerAndFit(sizer);

    openButton->Bind(wxEVT_BUTTON, &GraphPanel::OpenFromFile, this);
    saveButton->Bind(wxEVT_BUTTON, &GraphPanel::SaveToFile, this);
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
                fileOpenCallback();
			} catch (const core::graph6FormatError& err) {
                wxMessageBox("Could not open file " + fileDialog->GetFilename() + "\nError: " + err.what());
			}
		}
	}

	fileDialog->Destroy();
}

void GraphPanel::SaveToFile(wxCommandEvent& event) {
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
                wxMessageBox("Could save to file " + fileDialog->GetFilename() + "\nError: " + err.what());
			}
		}
	}

	fileDialog->Destroy();
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
