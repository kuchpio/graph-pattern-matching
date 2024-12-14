#pragma once

#include "wx/glcanvas.h"

class GraphCanvas : public wxGLCanvas {
    const char* nodeVertexShaderPath = "node.vsh";
    const char* nodeFragmentShaderPath = "node.fsh";
    const char* edgeVertexShaderPath = "edge.vsh";
    const char* edgeFragmentShaderPath = "edge.fsh";

    const float *backgroundColor;
    const float lightModeBackground[4] = {0.9, 0.9, 0.9, 1.0};
    const float lightModeColors[4 * (4 + 9)] = {
		0.1, 0.1, 0.1, 1.0,
		0.1, 0.1, 0.48, 1.0,
		0.48, 0.1, 0.1, 1.0,
		0.29, 0.1, 0.29, 1.0,
		0.75, 0.75, 0.75, 1.0,
		0.80, 0.31, 0.26, 1.0,
		0.39, 0.67, 0.28, 1.0,
		0.64, 0.38, 0.78, 1.0,
		0.60, 0.59, 0.25, 1.0,
		0.40, 0.53, 0.80, 1.0,
		0.79, 0.52, 0.26, 1.0,
		0.29, 0.67, 0.55, 1.0,
		0.78, 0.36, 0.54, 1.0,
    };
    const float darkModeBackground[4] = {0.1, 0.1, 0.1, 1.0};
    const float darkModeColors[4 * (4 + 9)] = {
		0.85, 0.85, 0.85, 1.0,
		0.75, 0.75, 1.0, 1.0,
		1.0, 0.50, 0.50, 1.0,
		0.87, 0.62, 0.75, 1.0,
		0.25, 0.25, 0.25, 1.0,
		0.80, 0.31, 0.26, 1.0,
		0.39, 0.67, 0.28, 1.0,
		0.64, 0.38, 0.78, 1.0,
		0.60, 0.59, 0.25, 1.0,
		0.40, 0.53, 0.80, 1.0,
		0.79, 0.52, 0.26, 1.0,
		0.29, 0.67, 0.55, 1.0,
		0.78, 0.36, 0.54, 1.0,
    };

    wxGLContext* openGLContext;
    bool isOpenGLInitialized = false, isOpenGLInitializationAttempted = false;
    wxSize viewPortSize;

    unsigned int vertexArrayObject = 0;
    unsigned int vertexBuffer = 0;
    unsigned int vertexStateBuffer = 0;
    unsigned int vertexLabelsBuffer = 0;
    unsigned int edgesBuffer = 0;
    unsigned int nodeShaderProgram = 0;
    unsigned int edgeShaderProgram = 0;
    unsigned int settingsUniformBufferObject = 0;
    unsigned int colorsUniformBufferObject = 0;

    unsigned int vertexCount = 0;
    unsigned int edgesCount = 0;

    bool InitializeOpenGLFunctions();
    bool InitializeOpenGL();
    std::optional<unsigned int> InitializeShader(const char* vertexShaderPath, const char* fragmentShaderPath);
    void SetNodeSize(float radius, float border) const;

    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);
    void UpdateCanvasSize() const;

  public:
    const float NODE_RADIUS = 25.0;
    const float NODE_BORDER = 4.0;
    const float EDGE_WIDTH = 2.0;

    GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs);
    ~GraphCanvas();

    void SetVertexPositions(const float* positions2D, unsigned int vertexCount);
    void SetVertexStates(const unsigned int* states, unsigned int vertexCount);
    void SetVertexLabels(const unsigned int* labels, unsigned int vertexCount);
    void SetEdges(const unsigned int* edges, unsigned int edgesCount);
    const std::pair<int, int> CanvasSize() const;
    void SetBoundingSize(float width, float height) const;
    void SetCenterPosition(float x, float y) const;
};
