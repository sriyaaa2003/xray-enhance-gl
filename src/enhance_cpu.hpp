#pragma once
// Plain C++ reference versions of every shader. They exist to check the GPU path, not to be fast.
#include <cstdint>
#include <vector>

#include "image.hpp"
#include "lut.hpp"

namespace xe::cpu {

enum class Curve { Linear = 0, Sigmoid = 1 };

Image windowLevel(const Image& in, Window window, Curve curve = Curve::Linear);
std::vector<uint32_t> histogram(const Image& in, int bins, int tilesX = 1, int tilesY = 1);
Image applyLut(const Image& in, const std::vector<float>& lut);
Image equalise(const Image& in, int bins = 256);
Image matchHistogram(const Image& in, const Image& reference, int bins = 256);
Image clahe(const Image& in, int tiles, float clipLimit, int bins = 256);
Image gaussian(const Image& in, float sigma);
Image bilateral(const Image& in, float sigmaSpace, float sigmaRange);
Image unsharp(const Image& in, float sigma, float amount);
Image localContrast(const Image& in, int radius, float strength, float maxGain);

float standardDeviation(const Image& in);
int gaussianRadius(float sigma);
int bilateralRadius(float sigmaSpace);

}  // namespace xe::cpu
