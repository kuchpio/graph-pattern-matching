#include "glad/glad.h"
#include "wx/msgdlg.h"

#include "graphCanvas.h"

GraphCanvas::GraphCanvas(wxWindow* parent, const wxGLAttributes& canvasAttrs) : wxGLCanvas(parent, canvasAttrs) {
    wxGLContextAttrs ctxAttrs;
    ctxAttrs.PlatformDefaults().CoreProfile().OGLVersion(3, 3).EndList();
    openGLContext = new wxGLContext(this, nullptr, &ctxAttrs);

    if (!openGLContext->IsOK()) {
        wxMessageBox("This sample needs an OpenGL 3.3 capable driver.", "OpenGL version error", 
            wxOK | wxICON_INFORMATION, this);
        delete openGLContext;
        openGLContext = nullptr;
    }

    Bind(wxEVT_PAINT, &GraphCanvas::OnPaint, this);
    Bind(wxEVT_SIZE, &GraphCanvas::OnSize, this);
}

GraphCanvas::~GraphCanvas() {
    delete openGLContext;
}

bool GraphCanvas::InitializeOpenGLFunctions() {
    auto gladVersion = gladLoadGL();

    if (0 == gladVersion) {
        wxLogError("OpenGL glad initialization failed");
        return false;
    }

    wxLogDebug("Status: Using Glad");

    return true;
}

bool GraphCanvas::InitializeOpenGL() {
    if (!openGLContext) return false;

    SetCurrent(*openGLContext);

    if (!InitializeOpenGLFunctions()) {
        wxMessageBox("Error: Could not initialize OpenGL function pointers.",
                     "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    wxLogDebug("OpenGL version: %s", reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    wxLogDebug("OpenGL vendor: %s", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

    if (!InitializeShaders()) {
        wxMessageBox("Error: Could not initialize OpenGL shaders.", 
                     "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    glGenVertexArrays(1, &vertexArrayObject);
    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &edgesBuffer);

    glPointSize(VERTEX_SIZE);
    glLineWidth(EDGE_WIDTH);

    return true;
}

void GraphCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    if (!isOpenGLInitialized) return;

    SetCurrent(*openGLContext);
    glClearColor(0.9f, 0.9f, 0.9f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    glUniform4f(
        glGetUniformLocation(shaderProgram, "vertexColor"), 
        vertexColor.Red() / 255.0f, 
        vertexColor.Green() / 255.0f,
        vertexColor.Blue() / 255.0f, 
        1.0f
    );

	glBindVertexArray(vertexArrayObject);

	glDrawElements(GL_LINES, edgesCount * 2, GL_UNSIGNED_INT, 0);
	glDrawArrays(GL_POINTS, 0, vertexCount);

	glBindVertexArray(0);

    SwapBuffers();
}

void GraphCanvas::OnSize(wxSizeEvent& event) {
    bool firstApperance = IsShownOnScreen() && !isOpenGLInitialized;

    if (firstApperance)
        isOpenGLInitialized = InitializeOpenGL();

    if (isOpenGLInitialized) {
		auto viewPortSize = event.GetSize() * GetContentScaleFactor();
        SetCurrent(*openGLContext);
		glViewport(0, 0, viewPortSize.x, viewPortSize.y);
    }

    event.Skip();
}

bool GraphCanvas::InitializeShaders() {
    constexpr auto vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        void main()
        {
            gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
        }
    )";

    constexpr auto fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        uniform vec4 vertexColor;
        void main()
        {
            FragColor = vertexColor;
        }
    )";

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        wxLogDebug("Vertex Shader Compilation Failed: %s", infoLog);
        return false;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        wxLogDebug("Fragment Shader Compilation Failed: %s", infoLog);
        return false;
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        wxLogDebug("Shader Program Linking Failed: %s", infoLog);
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return true;
}

void GraphCanvas::SetVertexPositions(const float* positions2D, unsigned int vertexCount) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    glBindVertexArray(vertexArrayObject);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 2, positions2D, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    this->vertexCount = vertexCount;
}

void GraphCanvas::SetEdges(const unsigned int* edges, unsigned int edgesCount) {
    if (!isOpenGLInitialized) return;
    SetCurrent(*openGLContext);

    glBindVertexArray(vertexArrayObject);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, edgesBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * edgesCount * 2, edges, GL_STATIC_DRAW);

    glBindVertexArray(0);

    this->edgesCount = edgesCount;
}
