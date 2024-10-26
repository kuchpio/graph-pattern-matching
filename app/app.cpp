#include <iostream>

#include "glad/glad.h"

#include "wx/wx.h"
#include "wx/glcanvas.h"
#include "wx/colordlg.h"

#include "image.h"
#include "pattern.h"

class App : public wxApp {
  public:
    bool OnInit() override;
};

class OpenGLCanvas;

class Frame : public wxFrame {
  public:
    Frame(const wxString& title);

  private:
    OpenGLCanvas* openGLCanvas = nullptr;
};

class OpenGLCanvas : public wxGLCanvas {
  public:
    OpenGLCanvas(Frame* parent, const wxGLAttributes& canvasAttrs);
    ~OpenGLCanvas();

    bool InitializeOpenGLFunctions();
    bool InitializeOpenGL();

    void OnPaint(wxPaintEvent& event);
    void OnSize(wxSizeEvent& event);

    wxColour triangleColor{wxColour(255, 128, 51)};

  private:
    wxGLContext* openGLContext;
    bool isOpenGLInitialized{false};

    unsigned int VAO = 0;
    unsigned int VBO = 0;
    unsigned int shaderProgram = 0;
};

bool App::OnInit() {
    auto frame = new Frame("Graph pattern matching");
    frame->Show();
    return true;
}

Frame::Frame(const wxString& title) : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxDefaultSize) {
    auto sizer = new wxBoxSizer(wxVERTICAL);

    wxGLAttributes vAttrs;
    vAttrs.PlatformDefaults().Defaults().EndList();

    if (wxGLCanvas::IsDisplaySupported(vAttrs)) {
        openGLCanvas = new OpenGLCanvas(this, vAttrs);
        openGLCanvas->SetMinSize(FromDIP(wxSize(640, 480)));
        sizer->Add(openGLCanvas, 1, wxEXPAND);
    }

    auto bottomSizer = new wxBoxSizer(wxHORIZONTAL);
    auto colorButton = new wxButton(this, wxID_ANY, "Change Color");

    bottomSizer->Add(colorButton, 0, wxALL | wxALIGN_CENTER, FromDIP(15));
    bottomSizer->AddStretchSpacer(1);

    sizer->Add(bottomSizer, 0, wxEXPAND);

    SetSizerAndFit(sizer);

    colorButton->Bind(wxEVT_BUTTON, [this](wxCommandEvent& event) {
        wxColourData colorData;
        colorData.SetColour(this->openGLCanvas->triangleColor);
        wxColourDialog dialog(this, &colorData);

        if (dialog.ShowModal() == wxID_OK) {
            openGLCanvas->triangleColor = dialog.GetColourData().GetColour();
            openGLCanvas->Refresh();
        }
    });
}

OpenGLCanvas::OpenGLCanvas(Frame* parent, const wxGLAttributes& canvasAttrs) : wxGLCanvas(parent, canvasAttrs) {
    wxGLContextAttrs ctxAttrs;
    ctxAttrs.PlatformDefaults().CoreProfile().OGLVersion(3, 3).EndList();
    openGLContext = new wxGLContext(this, nullptr, &ctxAttrs);

    if (!openGLContext->IsOK()) {
        wxMessageBox("This sample needs an OpenGL 3.3 capable driver.", "OpenGL version error",
                     wxOK | wxICON_INFORMATION, this);
        delete openGLContext;
        openGLContext = nullptr;
    }

    Bind(wxEVT_PAINT, &OpenGLCanvas::OnPaint, this);
    Bind(wxEVT_SIZE, &OpenGLCanvas::OnSize, this);
}

OpenGLCanvas::~OpenGLCanvas() {
    delete openGLContext;
}

bool OpenGLCanvas::InitializeOpenGLFunctions() {
    auto gladVersion = gladLoadGL();

    if (0 == gladVersion) {
        wxLogError("OpenGL glad initialization failed");
        return false;
    }

    wxLogDebug("Status: Using Glad");

    return true;
}

bool OpenGLCanvas::InitializeOpenGL() {
    if (!openGLContext) {
        return false;
    }

    SetCurrent(*openGLContext);

    if (!InitializeOpenGLFunctions()) {
        wxMessageBox("Error: Could not initialize OpenGL function pointers.", "OpenGL initialization error",
                     wxOK | wxICON_INFORMATION, this);
        return false;
    }

    wxLogDebug("OpenGL version: %s", reinterpret_cast<const char*>(glGetString(GL_VERSION)));
    wxLogDebug("OpenGL vendor: %s", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));

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
        uniform vec4 triangleColor;
        void main()
        {
            FragColor = triangleColor;
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
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        wxLogDebug("Fragment Shader Compilation Failed: %s", infoLog);
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        wxLogDebug("Shader Program Linking Failed: %s", infoLog);
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    float vertices[] = {-0.5f, -0.5f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f};

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    isOpenGLInitialized = true;

    return true;
}

void OpenGLCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)) {
    wxPaintDC dc(this);

    if (!isOpenGLInitialized) {
        return;
    }

    SetCurrent(*openGLContext);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    int colorLocation = glGetUniformLocation(shaderProgram, "triangleColor");
    glUniform4f(colorLocation, triangleColor.Red() / 255.0f, triangleColor.Green() / 255.0f,
                triangleColor.Blue() / 255.0f, 1.0f);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    SwapBuffers();
}

void OpenGLCanvas::OnSize(wxSizeEvent& event) {
    bool firstApperance = IsShownOnScreen() && !isOpenGLInitialized;

    if (firstApperance) {
        InitializeOpenGL();
    }

    if (isOpenGLInitialized) {
        auto viewPortSize = event.GetSize() * GetContentScaleFactor();
        glViewport(0, 0, viewPortSize.x, viewPortSize.y);
    }

    event.Skip();
}

wxIMPLEMENT_APP_NO_MAIN(App);

int main() {
    auto bigGraph = image::grapherize(8);
    auto smallGraph = image::grapherize(6);

    if (pattern::match(bigGraph, smallGraph)) {
        std::cout << "Match found." << std::endl;
    } else {
        std::cout << "Match not found." << std::endl;
    }

    return wxEntry();
}
