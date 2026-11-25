#pragma once
// A very small OpenGL layer: headless context, shader programs, float textures and a full-screen pass.
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/glcorearb.h>

#include <cstdint>
#include <string>
#include <vector>

#include "image.hpp"

namespace xe {

// OpenGL 4.3 core context created through EGL, so no window or display server is needed.
class HeadlessContext {
public:
    HeadlessContext();
    ~HeadlessContext();
    HeadlessContext(const HeadlessContext&) = delete;
    HeadlessContext& operator=(const HeadlessContext&) = delete;
    std::string renderer() const;
    std::string version() const;

private:
    void* display_ = nullptr;
    void* context_ = nullptr;
};

void checkGl(const char* where);

class Program {
public:
    static Program fragment(const std::string& fragmentFile);  // paired with fullscreen.vert
    static Program compute(const std::string& computeFile);
    Program(Program&& other) noexcept;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    ~Program();

    GLuint id() const { return id_; }
    void set(const char* name, int value) const;
    void set(const char* name, float value) const;
    void set(const char* name, int x, int y) const;

private:
    explicit Program(GLuint id) : id_(id) {}
    GLint location(const char* name) const;
    GLuint id_ = 0;
};

// Single-channel 32-bit float 2D texture.
class Texture {
public:
    Texture(int width, int height, const float* data = nullptr);
    explicit Texture(const Image& image) : Texture(image.width, image.height, image.pixels.data()) {}
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;
    ~Texture();

    GLuint id() const { return id_; }
    int width() const { return width_; }
    int height() const { return height_; }
    void bind(int unit) const;
    Image read() const;

private:
    GLuint id_ = 0;
    int width_ = 0;
    int height_ = 0;
};

// Runs a fragment shader once per pixel of the target texture.
class FullscreenPass {
public:
    FullscreenPass();
    FullscreenPass(const FullscreenPass&) = delete;
    FullscreenPass& operator=(const FullscreenPass&) = delete;
    ~FullscreenPass();
    void run(const Program& program, const Texture& target) const;

private:
    GLuint framebuffer_ = 0;
    GLuint vertexArray_ = 0;
};

}  // namespace xe
