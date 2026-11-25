#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>

#include "enhance_cpu.hpp"
#include "enhance_gpu.hpp"
#include "gl.hpp"
#include "metrics.hpp"

using namespace xe;

namespace {

const gpu::Enhancer* g_gpu = nullptr;

// A gradient with a bright disc, a dark box and noise. The size is deliberately not a multiple of 8.
Image testImage(int width = 173, int height = 131, unsigned seed = 7) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> noise(-0.05f, 0.05f);
    Image image(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            float v = 0.2f + 0.4f * x / width;
            if (std::hypot(x - width * 0.35f, y - height * 0.4f) < 25) v += 0.3f;
            if (x > 100 && x < 140 && y > 60 && y < 100) v -= 0.15f;
            image.at(x, y) = std::fmin(std::fmax(v + noise(rng), 0.0f), 1.0f);
        }
    return image;
}

float maxDifference(const Image& a, const Image& b) {
    REQUIRE(a.width == b.width);
    REQUIRE(a.height == b.height);
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::fmax(worst, std::fabs(a.pixels[i] - b.pixels[i]));
    return worst;
}

// Largest allowed GPU-minus-CPU difference on a 0-to-1 scale. 1e-4 is far below one grey level in 255.
constexpr float kTolerance = 1e-4f;

void checkAgainstCpu(const char* name, const Image& fromGpu, const Image& fromCpu) {
    const float worst = maxDifference(fromGpu, fromCpu);
    std::printf("  %-28s largest GPU-CPU difference %.2e\n", name, worst);
    CHECK(worst <= kTolerance);
}

}  // namespace

TEST_CASE("PFM files keep pixel values and row order") {
    const Image image = testImage(31, 17);
    writePfm("roundtrip.pfm", image);
    const Image back = readPfm("roundtrip.pfm");
    CHECK(maxDifference(image, back) == 0.0f);
}

TEST_CASE("compute-shader histogram counts exactly what the CPU counts") {
    const Image image = testImage();
    for (int bins : {64, 256, 4096}) {
        const auto fromGpu = g_gpu->histogram(image, bins);
        CHECK(fromGpu == cpu::histogram(image, bins));
        CHECK(std::accumulate(fromGpu.begin(), fromGpu.end(), size_t{0}) == image.size());
    }
    CHECK(g_gpu->histogram(image, 256, 8, 8) == cpu::histogram(image, 256, 8, 8));
}

TEST_CASE("point operations match the CPU reference") {
    const Image image = testImage();
    const Image reference = testImage(173, 131, 99);
    checkAgainstCpu("window/level linear", g_gpu->windowLevel(image, {0.45f, 0.3f}),
                    cpu::windowLevel(image, {0.45f, 0.3f}));
    checkAgainstCpu("window/level sigmoid", g_gpu->windowLevel(image, {0.45f, 0.3f}, cpu::Curve::Sigmoid),
                    cpu::windowLevel(image, {0.45f, 0.3f}, cpu::Curve::Sigmoid));
    checkAgainstCpu("histogram equalisation", g_gpu->equalise(image), cpu::equalise(image));
    checkAgainstCpu("histogram matching", g_gpu->matchHistogram(image, reference),
                    cpu::matchHistogram(image, reference));
    checkAgainstCpu("CLAHE", g_gpu->clahe(image, 8, 2.0f), cpu::clahe(image, 8, 2.0f));
}

TEST_CASE("neighbourhood filters match the CPU reference") {
    const Image image = testImage();
    checkAgainstCpu("gaussian", g_gpu->gaussian(image, 1.6f), cpu::gaussian(image, 1.6f));
    checkAgainstCpu("bilateral", g_gpu->bilateral(image, 2.0f, 0.1f), cpu::bilateral(image, 2.0f, 0.1f));
    checkAgainstCpu("unsharp mask", g_gpu->unsharp(image, 1.5f, 1.0f), cpu::unsharp(image, 1.5f, 1.0f));
    checkAgainstCpu("local contrast", g_gpu->localContrast(image, 7, 0.8f, 3.0f),
                    cpu::localContrast(image, 7, 0.8f, 3.0f));
}

TEST_CASE("behaviour that must hold whatever the implementation") {
    const Image image = testImage();
    const Image flat(40, 30, 0.37f);
    CHECK(maxDifference(g_gpu->gaussian(flat, 2.0f), flat) < 1e-6f);
    CHECK(maxDifference(g_gpu->bilateral(flat, 2.0f, 0.05f), flat) < 1e-6f);
    CHECK(maxDifference(g_gpu->unsharp(image, 1.5f, 0.0f), image) < 1e-6f);
    // A window that spans the whole range changes nothing.
    CHECK(maxDifference(g_gpu->windowLevel(image, {0.5f, 1.0f}), image) < 1e-6f);
    // Matching an image to itself moves no pixel by more than one bin.
    CHECK(maxDifference(g_gpu->matchHistogram(image, image), image) <= 1.0f / 255.0f + 1e-6f);
}

TEST_CASE("CLAHE tables never decrease and end at 1") {
    const auto histograms = cpu::histogram(testImage(), 256, 4, 4);
    for (float clip : {1.0f, 2.0f, 40.0f}) {
        const auto luts = claheLuts(histograms, 256, 16, clip);
        for (int tile = 0; tile < 16; ++tile) {
            for (int b = 1; b < 256; ++b) CHECK(luts[tile * 256 + b] >= luts[tile * 256 + b - 1]);
            CHECK(luts[tile * 256 + 255] == doctest::Approx(1.0f));
        }
    }
}

TEST_CASE("percentile window covers the chosen share of pixels") {
    Image ramp(1001, 1);
    for (int x = 0; x <= 1000; ++x) ramp.at(x, 0) = x / 1000.0f;
    const Window window = g_gpu->autoWindow(ramp, 10.0f, 90.0f);
    CHECK(window.level == doctest::Approx(0.5f).epsilon(0.01));
    CHECK(window.width == doctest::Approx(0.8f).epsilon(0.01));
}

TEST_CASE("study measures behave as intended") {
    Image mask(96, 96), clean(96, 96, 0.3f);
    for (int y = 0; y < 96; ++y)
        for (int x = 40; x < 96; ++x) {
            mask.at(x, y) = 1.0f;
            clean.at(x, y) = 0.6f;
        }
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 0.03f);
    Image noisy = clean;
    for (float& v : noisy.pixels) v += noise(rng);
    const Image edge = metrics::boundary(mask, 2);
    const Image everywhere(96, 96, 1.0f);

    CHECK(metrics::ssim(clean, clean) == doctest::Approx(1.0));
    CHECK(metrics::edgeRetention(clean, clean, edge) == doctest::Approx(1.0));
    CHECK(metrics::edgeRetention(cpu::gaussian(clean, 2.0f), clean, edge) < 0.9);
    CHECK(metrics::edgeRetention(cpu::unsharp(clean, 1.5f, 1.0f), clean, edge) > 1.1);
    CHECK(metrics::edgeRetention(noisy, clean, edge) == doctest::Approx(1.0).epsilon(0.1));
    CHECK(metrics::remainingNoise(noisy, clean, everywhere) == doctest::Approx(0.03).epsilon(0.1));
    const metrics::BoundaryStats before = metrics::boundaryStats(noisy, mask);
    CHECK(before.contrast == doctest::Approx(0.3).epsilon(0.05));
    CHECK(before.cnr == doctest::Approx(10.0).epsilon(0.2));
    // A linear intensity change without clipping changes the visible step but not contrast-to-noise.
    Image scaled = noisy;
    for (float& v : scaled.pixels) v = 0.5f * v + 0.1f;
    const metrics::BoundaryStats after = metrics::boundaryStats(scaled, mask);
    CHECK(after.contrast == doctest::Approx(0.5 * before.contrast).epsilon(0.001));
    CHECK(after.cnr == doctest::Approx(before.cnr).epsilon(0.001));
}

int main(int argc, char** argv) {
    const HeadlessContext context;
    std::printf("OpenGL renderer: %s\nOpenGL version:  %s\n", context.renderer().c_str(), context.version().c_str());
    const gpu::Enhancer enhancer;
    g_gpu = &enhancer;
    doctest::Context doctestContext(argc, argv);
    return doctestContext.run();
}
