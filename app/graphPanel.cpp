#include "wx/glcanvas.h"
#include "wx/colordlg.h"

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
    auto initButton = new wxButton(this, wxID_ANY, "Utwórz");
    auto colorButton = new wxButton(this, wxID_ANY, "Kolor");

    bottomSizer->Add(initButton, 0, wxALL | wxALIGN_CENTER, FromDIP(15));
    bottomSizer->Add(colorButton, 0, wxALL | wxALIGN_CENTER, FromDIP(15));
    bottomSizer->AddStretchSpacer(1);

    sizer->Add(bottomSizer, 0, wxEXPAND);

    this->SetSizerAndFit(sizer);

    initButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
		const float vertices[] = {
			-0.5f, 0.3f, 
			0.4f, 0.1f, 
			-0.7f, 0.9f, 
			-0.6f, 0.0f, 
			0.2f, -0.5f
		};
		const unsigned int edges[] = {
			0, 1, 
			0, 3, 
			0, 4, 
			1, 2, 
			2, 3, 
			2, 4
		};

        canvas->SetVertexPositions(vertices, 5);
        canvas->SetEdges(edges, 6);
		canvas->Refresh();
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
}