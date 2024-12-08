#pragma once

#include "wx/glcanvas.h"

class GraphCanvas : public wxGLCanvas {
    static char *nodeVertexShaderSource, *nodeFragmentShaderSource, *edgeVertexShaderSource, *edgeFragmentShaderSource;

    wxGLContext* openGLContext;
    bool isOpenGLInitialized = false, isOpenGLInitializationAttempted = false;

    unsigned int vertexArrayObject = 0;
    unsigned int vertexBuffer = 0;
    unsigned int vertexStateBuffer = 0;
    unsigned int edgesBuffer = 0;
    unsigned int nodeShaderProgram = 0;
    unsigned int edgeShaderProgram = 0;
    unsigned int settingsUniformBufferObject = 0;

    unsigned int vertexCount = 0;
    unsigned int edgesCount = 0;

    bool InitializeOpenGLFunctions();
    bool InitializeOpenGL();
    std::optional<unsigned int> GraphCanvas::InitializeShader(const char* vertexShaderSource,
                                                              const char* fragmentShaderSource);
    void SetCanvasSize(int width, int height) const;
    void SetBoundingSize(float width, float height) const;
    void SetCenterPosition(float x, float y) const;
    void SetNodeSize(float radius, float border) const;

    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);

  public:
    GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs);
    ~GraphCanvas();

    void SetVertexPositions(const float* positions2D, unsigned int vertexCount);
    void GraphCanvas::SetVertexStates(const unsigned int* states, unsigned int vertexCount);
    void SetEdges(const unsigned int* edges, unsigned int edgesCount);
};
