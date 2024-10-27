#pragma once

#include "wx/glcanvas.h"

class GraphCanvas : public wxGLCanvas {
  public:
    GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs);
    ~GraphCanvas();

    wxColour vertexColor{wxColour(255, 128, 51)};

    void SetVertexPositions(const float *positions2D, unsigned int vertexCount);
    void SetEdges(const unsigned int* edges, unsigned int edgesCount);

  private:
    wxGLContext* openGLContext;
    bool isOpenGLInitialized = false;

    const float VERTEX_SIZE = 8.0f;
    const float EDGE_WIDTH = 2.0f;

    unsigned int vertexArrayObject = 0;
    unsigned int vertexBuffer = 0;
    unsigned int edgesBuffer = 0;
    unsigned int shaderProgram = 0;

    unsigned int vertexCount = 0;
    unsigned int edgesCount = 0;

    bool InitializeOpenGLFunctions();
    bool InitializeOpenGL();
    bool InitializeShaders();

    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
};
