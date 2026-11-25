#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "enhance_cpu.hpp"

namespace xe::metrics {

namespace {

void requireSameSize(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height) throw std::invalid_argument("images differ in size");
}

Image product(const Image& a, const Image& b) {
    Image out(a.width, a.height);
    for (size_t i = 0; i < a.size(); ++i) out.pixels[i] = a.pixels[i] * b.pixels[i];
    return out;
}

Image morphology(const Image& mask, int radius, bool grow) {
    Image out(mask.width, mask.height);
    for (int y = 0; y < mask.height; ++y)
        for (int x = 0; x < mask.width; ++x) {
            bool result = !grow;
            for (int dy = -radius; dy <= radius; ++dy)
                for (int dx = -radius; dx <= radius; ++dx) {
                    const bool on = mask.clamped(x + dx, y + dy) > 0.5f;
                    result = grow ? (result || on) : (result && on);
                }
            out.at(x, y) = result ? 1.0f : 0.0f;
        }
    return out;
}

struct Gradient {
    float x, y;
};

Gradient sobel(const Image& in, int x, int y) {
    auto p = [&](int dx, int dy) { return in.clamped(x + dx, y + dy); };
    return {(p(1, -1) + 2 * p(1, 0) + p(1, 1) - p(-1, -1) - 2 * p(-1, 0) - p(-1, 1)) / 8.0f,
            (p(-1, 1) + 2 * p(0, 1) + p(1, 1) - p(-1, -1) - 2 * p(0, -1) - p(1, -1)) / 8.0f};
}

}  // namespace

double psnr(const Image& a, const Image& b) {
    requireSameSize(a, b);
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a.pixels[i] - b.pixels[i];
        sum += d * d;
    }
    const double mse = sum / a.size();
    return mse <= 1e-12 ? 120.0 : -10.0 * std::log10(mse);
}

double ssim(const Image& a, const Image& b) {
    requireSameSize(a, b);
    constexpr float sigma = 1.5f;
    const Image meanA = cpu::gaussian(a, sigma), meanB = cpu::gaussian(b, sigma);
    const Image squareA = cpu::gaussian(product(a, a), sigma), squareB = cpu::gaussian(product(b, b), sigma);
    const Image cross = cpu::gaussian(product(a, b), sigma);
    constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double ma = meanA.pixels[i], mb = meanB.pixels[i];
        const double va = squareA.pixels[i] - ma * ma, vb = squareB.pixels[i] - mb * mb;
        const double covariance = cross.pixels[i] - ma * mb;
        sum += ((2 * ma * mb + c1) * (2 * covariance + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2));
    }
    return sum / a.size();
}

Image dilate(const Image& mask, int radius) { return morphology(mask, radius, true); }
Image erode(const Image& mask, int radius) { return morphology(mask, radius, false); }

Image boundary(const Image& mask, int radius) {
    const Image grown = dilate(mask, radius), shrunk = erode(mask, radius);
    Image out(mask.width, mask.height);
    for (size_t i = 0; i < mask.size(); ++i) out.pixels[i] = grown.pixels[i] > 0.5f && shrunk.pixels[i] < 0.5f;
    return out;
}

double remainingNoise(const Image& filteredNoisy, const Image& filteredClean, const Image& region) {
    requireSameSize(filteredNoisy, filteredClean);
    requireSameSize(filteredNoisy, region);
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < region.size(); ++i)
        if (region.pixels[i] > 0.5f) {
            const double d = filteredNoisy.pixels[i] - filteredClean.pixels[i];
            sum += d * d;
            ++count;
        }
    return count ? std::sqrt(sum / count) : 0.0;
}

double edgeRetention(const Image& processed, const Image& clean, const Image& edgePixels) {
    requireSameSize(processed, clean);
    requireSameSize(processed, edgePixels);
    double kept = 0.0, total = 0.0;
    for (int y = 0; y < clean.height; ++y)
        for (int x = 0; x < clean.width; ++x) {
            if (edgePixels.at(x, y) < 0.5f) continue;
            const Gradient c = sobel(clean, x, y), p = sobel(processed, x, y);
            const double strength = std::hypot(c.x, c.y);
            if (strength < 1e-6) continue;
            kept += (p.x * c.x + p.y * c.y) / strength;
            total += strength;
        }
    return total > 0.0 ? kept / total : 0.0;
}

BoundaryStats boundaryStats(const Image& image, const Image& mask, int band, int tile, int minPixels) {
    requireSameSize(image, mask);
    const Image grown = dilate(mask, band), shrunk = erode(mask, band);
    double steps = 0.0, ratios = 0.0;
    int tiles = 0;
    for (int ty = 0; ty < image.height; ty += tile)
        for (int tx = 0; tx < image.width; tx += tile) {
            double sum[2] = {0, 0}, squares[2] = {0, 0};
            int count[2] = {0, 0};
            for (int y = ty; y < std::min(ty + tile, image.height); ++y)
                for (int x = tx; x < std::min(tx + tile, image.width); ++x) {
                    const bool inside = mask.at(x, y) > 0.5f && shrunk.at(x, y) < 0.5f;
                    const bool outside = mask.at(x, y) < 0.5f && grown.at(x, y) > 0.5f;
                    if (!inside && !outside) continue;
                    const int side = inside ? 1 : 0;
                    const double v = image.at(x, y);
                    sum[side] += v;
                    squares[side] += v * v;
                    ++count[side];
                }
            if (count[0] < minPixels || count[1] < minPixels) continue;
            const double meanOut = sum[0] / count[0], meanIn = sum[1] / count[1];
            const double varOut = std::max(squares[0] / count[0] - meanOut * meanOut, 0.0);
            const double varIn = std::max(squares[1] / count[1] - meanIn * meanIn, 0.0);
            steps += std::abs(meanIn - meanOut);
            ratios += std::abs(meanIn - meanOut) / std::sqrt(0.5 * (varIn + varOut) + 1e-8);
            ++tiles;
        }
    if (!tiles) return {0.0, 0.0};
    return {steps / tiles, ratios / tiles};
}

double clippedFraction(const Image& image, const Image& region) {
    requireSameSize(image, region);
    size_t clipped = 0, count = 0;
    for (size_t i = 0; i < image.size(); ++i)
        if (region.pixels[i] > 0.5f) {
            ++count;
            clipped += image.pixels[i] <= 0.0f || image.pixels[i] >= 1.0f;
        }
    return count ? static_cast<double>(clipped) / count : 0.0;
}

}  // namespace xe::metrics
