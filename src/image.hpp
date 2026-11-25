#pragma once
#include <algorithm>
#include <string>
#include <vector>

namespace xe {

// Single-channel float image, row-major. Display values are in [0, 1].
struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;

    Image() = default;
    Image(int w, int h, float value = 0.0f)
        : width(w), height(h), pixels(static_cast<size_t>(w) * static_cast<size_t>(h), value) {}

    size_t size() const { return pixels.size(); }
    float at(int x, int y) const { return pixels[static_cast<size_t>(y) * width + x]; }
    float& at(int x, int y) { return pixels[static_cast<size_t>(y) * width + x]; }
    // Edge pixels repeat outside the image (the shaders do the same).
    float clamped(int x, int y) const {
        return at(std::clamp(x, 0, width - 1), std::clamp(y, 0, height - 1));
    }
};

// Portable FloatMap, greyscale ("Pf"), little-endian. Rows are stored bottom to top on disk.
Image readPfm(const std::string& path);
void writePfm(const std::string& path, const Image& image);

}  // namespace xe
