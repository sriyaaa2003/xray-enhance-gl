#include "gl.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace xe {

HeadlessContext::HeadlessContext() {
    EGLDisplay display = EGL_NO_DISPLAY;
    auto getPlatformDisplay =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (getPlatformDisplay)
        display = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (display == EGL_NO_DISPLAY) display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr))
        throw std::runtime_error("EGL: no usable display");
    if (!eglBindAPI(EGL_OPENGL_API)) throw std::runtime_error("EGL: desktop OpenGL is not available");

    const EGLint attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 4,
                                 EGL_CONTEXT_MINOR_VERSION, 3,
                                 EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                 EGL_NONE};
    EGLContext context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, attributes);
    if (context == EGL_NO_CONTEXT) throw std::runtime_error("EGL: cannot create an OpenGL 4.3 core context");
    if (!eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
        throw std::runtime_error("EGL: cannot activate the context without a surface");
    display_ = display;
    context_ = context;
}

HeadlessContext::~HeadlessContext() {
    if (display_) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display_, context_);
        eglTerminate(display_);
    }
}

std::string HeadlessContext::renderer() const { return reinterpret_cast<const char*>(glGetString(GL_RENDERER)); }
std::string HeadlessContext::version() const { return reinterpret_cast<const char*>(glGetString(GL_VERSION)); }

void checkGl(const char* where) {
    const GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::ostringstream message;
        message << "OpenGL error 0x" << std::hex << error << " in " << where;
        throw std::runtime_error(message.str());
    }
}

namespace {

std::string readSource(const std::string& name) {
    // An absolute path is used as it is, so other projects can load their own shaders.
    const std::string path = !name.empty() && name.front() == '/' ? name : std::string(XENH_SHADER_DIR) + "/" + name;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open shader " + path);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

GLuint compileShader(GLenum type, const std::string& name) {
    const std::string source = readSource(name);
    const char* pointer = source.c_str();
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &pointer, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {0};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        throw std::runtime_error("shader " + name + " failed to compile:\n" + log);
    }
    return shader;
}

GLuint linkProgram(const std::vector<GLuint>& shaders, const std::string& name) {
    const GLuint program = glCreateProgram();
    for (GLuint shader : shaders) glAttachShader(program, shader);
    glLinkProgram(program);
    for (GLuint shader : shaders) glDeleteShader(shader);
    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {0};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        throw std::runtime_error("program " + name + " failed to link:\n" + log);
    }
    return program;
}

}  // namespace

Program Program::fragment(const std::string& fragmentFile) {
    return Program(linkProgram(
        {compileShader(GL_VERTEX_SHADER, "fullscreen.vert"), compileShader(GL_FRAGMENT_SHADER, fragmentFile)},
        fragmentFile));
}

Program Program::compute(const std::string& computeFile) {
    return Program(linkProgram({compileShader(GL_COMPUTE_SHADER, computeFile)}, computeFile));
}

Program::Program(Program&& other) noexcept : id_(other.id_) { other.id_ = 0; }

Program::~Program() {
    if (id_) glDeleteProgram(id_);
}

GLint Program::location(const char* name) const {
    const GLint where = glGetUniformLocation(id_, name);
    // A silent -1 would hide a misspelt uniform, so fail loudly instead.
    if (where < 0) throw std::runtime_error(std::string("unknown shader uniform: ") + name);
    return where;
}

void Program::set(const char* name, int value) const { glProgramUniform1i(id_, location(name), value); }
void Program::set(const char* name, float value) const { glProgramUniform1f(id_, location(name), value); }
void Program::set(const char* name, int x, int y) const { glProgramUniform2i(id_, location(name), x, y); }

Texture::Texture(int width, int height, const float* data) : width_(width), height_(height) {
    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (data) glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RED, GL_FLOAT, data);
    checkGl("Texture");
}

Texture::~Texture() {
    if (id_) glDeleteTextures(1, &id_);
}

void Texture::bind(int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id_);
}

Image Texture::read() const {
    Image image(width_, height_);
    glBindTexture(GL_TEXTURE_2D, id_);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, image.pixels.data());
    checkGl("Texture::read");
    return image;
}

FullscreenPass::FullscreenPass() {
    glGenFramebuffers(1, &framebuffer_);
    glGenVertexArrays(1, &vertexArray_);
}

FullscreenPass::~FullscreenPass() {
    glDeleteFramebuffers(1, &framebuffer_);
    glDeleteVertexArrays(1, &vertexArray_);
}

void FullscreenPass::run(const Program& program, const Texture& target) const {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target.id(), 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        throw std::runtime_error("float framebuffer is incomplete");
    glViewport(0, 0, target.width(), target.height());
    glUseProgram(program.id());
    glBindVertexArray(vertexArray_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    checkGl("FullscreenPass::run");
}

}  // namespace xe
