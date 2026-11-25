#pragma once
// The same operations as xe::cpu, run in GLSL. Per-pixel work is done in fragment shaders (the way a
// viewer would filter what it displays); histograms use a compute shader with atomic counters.
#include <cstdint>
#include <vector>

#include "enhance_cpu.hpp"
#include "gl.hpp"
#include "image.hpp"
#include "lut.hpp"

namespace xe::gpu {

using xe::cpu::Curve;

// Needs a current OpenGL context (see HeadlessContext).
class Enhancer {
public:
    Enhancer();

    Image windowLevel(const Image& in, Window window, Curve curve = Curve::Linear) const;
    std::vector<uint32_t> histogram(const Image& in, int bins, int tilesX = 1, int tilesY = 1) const;
    Window autoWindow(const Image& in, float lowPercent, float highPercent) const;
    Image applyLut(const Image& in, const std::vector<float>& lut) const;
    Image equalise(const Image& in, int bins = 256) const;
    Image matchHistogram(const Image& in, const Image& reference, int bins = 256) const;
    Image clahe(const Image& in, int tiles, float clipLimit, int bins = 256) const;
    Image gaussian(const Image& in, float sigma) const;
    Image bilateral(const Image& in, float sigmaSpace, float sigmaRange) const;
    Image unsharp(const Image& in, float sigma, float amount) const;
    Image localContrast(const Image& in, int radius, float strength, float maxGain) const;

private:
    FullscreenPass pass_;
    Program windowLevel_;
    Program histogram_;
    Program lut_;
    Program clahe_;
    Program gaussian_;
    Program bilateral_;
    Program unsharp_;
    Program localContrast_;
};

}  // namespace xe::gpu
