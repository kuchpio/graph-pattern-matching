#include "wx/glcanvas.h"
#include "wx/colordlg.h"
#include <chrono>

#include "graphPanel.h"
#include "graphCanvas.h"

GraphPanel::GraphPanel(wxWindow* parent) : wxPanel(parent) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        canvas = new GraphCanvas(this, vAttrs);
        canvas->SetMinSize(FromDIP(wxSize(640, 480)));
        sizer->Add(canvas, 1, wxEXPAND);
    }

    auto bottomSizer = new wxBoxSizer(wxHORIZONTAL);
    auto initButton = new wxButton(this, wxID_ANY, "Create");
    auto colorButton = new wxButton(this, wxID_ANY, "Color");

    bottomSizer->Add(initButton, 0, wxALL | wxALIGN_CENTER, FromDIP(15));
    bottomSizer->Add(colorButton, 0, wxALL | wxALIGN_CENTER, FromDIP(15));
    bottomSizer->AddStretchSpacer(1);

    sizer->Add(bottomSizer, 0, wxEXPAND);

    this->SetSizerAndFit(sizer);

    initButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) { 
        InitRandomGraph(5);
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
    if (adjecencyMatrix)       delete[] adjecencyMatrix;
    if (vertexPositions2D[0])  delete[] vertexPositions2D[0];
    if (vertexPositions2D[1])  delete[] vertexPositions2D[1];
    if (vertexVelocities2D[0]) delete[] vertexVelocities2D[0];
    if (vertexVelocities2D[1]) delete[] vertexVelocities2D[1];
}

void GraphPanel::InitRandomGraph(unsigned int vertexCount) {
    if (adjecencyMatrix)       delete[] adjecencyMatrix;
    if (vertexPositions2D[0])  delete[] vertexPositions2D[0];
    if (vertexPositions2D[1])  delete[] vertexPositions2D[1];
    if (vertexVelocities2D[0]) delete[] vertexVelocities2D[0];
    if (vertexVelocities2D[1]) delete[] vertexVelocities2D[1];

    adjecencyMatrix          = new bool[vertexCount * vertexCount];
    vertexPositions2D[0]     = new float[2 * vertexCount];
    vertexPositions2D[1]     = new float[2 * vertexCount];
    vertexVelocities2D[0]    = new float[2 * vertexCount];
    vertexVelocities2D[1]    = new float[2 * vertexCount];
    this->vertexCount = vertexCount;

    std::vector<unsigned int> edges;
    for (unsigned int i = 0; i < vertexCount; i++) {
        for (unsigned int j = 0; j < i; j++) {
            bool isEdge = rand() & 0b1;
            adjecencyMatrix[i * vertexCount + j] = adjecencyMatrix[j * vertexCount + i] = isEdge;
            if (isEdge) {
                edges.push_back(i);
                edges.push_back(j);
            }
        }
        vertexPositions2D[readBufferId][2 * i] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexPositions2D[readBufferId][2 * i + 1] = 2 * ((float)rand() / RAND_MAX) - 1;
        vertexVelocities2D[readBufferId][2 * i] = vertexVelocities2D[readBufferId][2 * i + 1] = 0.0f;
    }
    canvas->SetVertexPositions(vertexPositions2D[readBufferId], vertexCount);
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

    for (unsigned int i = 0; i < vertexCount; i++) {
        float x = vertexPositions2D[readBufferId][2 * i];
        float y = vertexPositions2D[readBufferId][2 * i + 1];

        float distOrigin = sqrtf(x * x + y * y);
        float gravityCoefficient = distOrigin < 0.1f ? 0.0f : C[3] / (distOrigin * distOrigin * distOrigin);

        float a_x = gravityCoefficient * x;
        float a_y = gravityCoefficient * y;

        for (unsigned int j = 0; j < vertexCount; j++) {
            float dx = x - vertexPositions2D[readBufferId][2 * j];
            float dy = y - vertexPositions2D[readBufferId][2 * j + 1];
            float dist = sqrtf(dx * dx + dy * dy);
            
            float springCoefficient = !adjecencyMatrix[i * vertexCount + j] || dist < 0.01f
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
    canvas->SetVertexPositions(vertexPositions2D[readBufferId], vertexCount);

    canvas->Refresh();
    event.RequestMore();
}
