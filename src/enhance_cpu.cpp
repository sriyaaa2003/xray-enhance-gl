#include "enhance_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace xe::cpu {

namespace {

float clamp01(float v) { return std::min(std::max(v, 0.0f), 1.0f); }
int binOf(float v, int bins) { return static_cast<int>(clamp01(v) * static_cast<float>(bins - 1) + 0.5f); }

Image blurAlong(const Image& in, float sigma, int radius, int dx, int dy) {
    Image out(in.width, in.height);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            float sum = 0.0f, weights = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const float w = std::exp(-static_cast<float>(k * k) / (2.0f * sigma * sigma));
                sum += w * in.clamped(x + k * dx, y + k * dy);
                weights += w;
            }
            out.at(x, y) = sum / weights;
        }
    return out;
}

}  // namespace

int gaussianRadius(float sigma) { return std::max(1, static_cast<int>(std::ceil(3.0f * sigma))); }
int bilateralRadius(float sigmaSpace) { return std::max(1, static_cast<int>(std::ceil(2.0f * sigmaSpace))); }

Image windowLevel(const Image& in, Window window, Curve curve) {
    Image out(in.width, in.height);
    const float width = std::max(window.width, 1e-6f);
    for (size_t i = 0; i < in.size(); ++i) {
        const float t = (in.pixels[i] - window.level) / width;
        out.pixels[i] = curve == Curve::Sigmoid ? 1.0f / (1.0f + std::exp(-4.0f * t)) : clamp01(t + 0.5f);
    }
    return out;
}

std::vector<uint32_t> histogram(const Image& in, int bins, int tilesX, int tilesY) {
    std::vector<uint32_t> counts(static_cast<size_t>(bins) * tilesX * tilesY, 0u);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            const int tileX = (x * tilesX) / in.width;
            const int tileY = (y * tilesY) / in.height;
            ++counts[static_cast<size_t>(tileY * tilesX + tileX) * bins + binOf(in.at(x, y), bins)];
        }
    return counts;
}

Image applyLut(const Image& in, const std::vector<float>& lut) {
    Image out(in.width, in.height);
    const int bins = static_cast<int>(lut.size());
    for (size_t i = 0; i < in.size(); ++i) out.pixels[i] = lut[binOf(in.pixels[i], bins)];
    return out;
}

Image equalise(const Image& in, int bins) { return applyLut(in, equalisationLut(histogram(in, bins))); }

Image matchHistogram(const Image& in, const Image& reference, int bins) {
    return applyLut(in, matchingLut(histogram(in, bins), histogram(reference, bins)));
}

Image clahe(const Image& in, int tiles, float clipLimit, int bins) {
    const std::vector<float> luts = claheLuts(histogram(in, bins, tiles, tiles), bins, tiles * tiles, clipLimit);
    Image out(in.width, in.height);
    const float tileWidth = static_cast<float>(in.width) / tiles;
    const float tileHeight = static_cast<float>(in.height) / tiles;
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            const int bin = binOf(in.at(x, y), bins);
            // Position in tile units, with tile centres on whole numbers.
            const float gx = (static_cast<float>(x) + 0.5f) / tileWidth - 0.5f;
            const float gy = (static_cast<float>(y) + 0.5f) / tileHeight - 0.5f;
            const int baseX = static_cast<int>(std::floor(gx));
            const int baseY = static_cast<int>(std::floor(gy));
            const float fx = gx - baseX, fy = gy - baseY;
            const int x0 = std::clamp(baseX, 0, tiles - 1), x1 = std::clamp(baseX + 1, 0, tiles - 1);
            const int y0 = std::clamp(baseY, 0, tiles - 1), y1 = std::clamp(baseY + 1, 0, tiles - 1);
            auto lut = [&](int tx, int ty) { return luts[static_cast<size_t>(ty * tiles + tx) * bins + bin]; };
            const float top = lut(x0, y0) * (1.0f - fx) + lut(x1, y0) * fx;
            const float bottom = lut(x0, y1) * (1.0f - fx) + lut(x1, y1) * fx;
            out.at(x, y) = top * (1.0f - fy) + bottom * fy;
        }
    return out;
}

Image gaussian(const Image& in, float sigma) {
    const int radius = gaussianRadius(sigma);
    return blurAlong(blurAlong(in, sigma, radius, 1, 0), sigma, radius, 0, 1);
}

Image bilateral(const Image& in, float sigmaSpace, float sigmaRange) {
    const int radius = bilateralRadius(sigmaSpace);
    Image out(in.width, in.height);
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            const float centre = in.at(x, y);
            float sum = 0.0f, weights = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    const float value = in.clamped(x + dx, y + dy);
                    const float difference = value - centre;
                    const float w = std::exp(-static_cast<float>(dx * dx + dy * dy) / (2.0f * sigmaSpace * sigmaSpace) -
                                             difference * difference / (2.0f * sigmaRange * sigmaRange));
                    sum += w * value;
                    weights += w;
                }
            out.at(x, y) = sum / weights;
        }
    return out;
}

Image unsharp(const Image& in, float sigma, float amount) {
    const Image blurred = gaussian(in, sigma);
    Image out(in.width, in.height);
    for (size_t i = 0; i < in.size(); ++i)
        out.pixels[i] = clamp01(in.pixels[i] + amount * (in.pixels[i] - blurred.pixels[i]));
    return out;
}

float standardDeviation(const Image& in) {
    double sum = 0.0, squares = 0.0;
    for (float v : in.pixels) {
        sum += v;
        squares += static_cast<double>(v) * v;
    }
    const double mean = sum / in.size();
    return static_cast<float>(std::sqrt(std::max(squares / in.size() - mean * mean, 0.0)));
}

Image localContrast(const Image& in, int radius, float strength, float maxGain) {
    const float globalDeviation = standardDeviation(in);
    Image out(in.width, in.height);
    const float count = static_cast<float>((2 * radius + 1) * (2 * radius + 1));
    for (int y = 0; y < in.height; ++y)
        for (int x = 0; x < in.width; ++x) {
            const float centre = in.at(x, y);
            // Sums of differences from the centre pixel keep 32-bit precision usable.
            float sum = 0.0f, squares = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    const float d = in.clamped(x + dx, y + dy) - centre;
                    sum += d;
                    squares += d * d;
                }
            const float meanDifference = sum / count;
            const float mean = centre + meanDifference;
            const float deviation = std::sqrt(std::max(squares / count - meanDifference * meanDifference, 0.0f));
            const float gain = std::min(std::max(strength * globalDeviation / (deviation + 1e-4f), 1.0f), maxGain);
            out.at(x, y) = clamp01(mean + gain * (centre - mean));
        }
    return out;
}

}  // namespace xe::cpu
