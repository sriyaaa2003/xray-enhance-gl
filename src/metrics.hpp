#pragma once
// Image measures used by the study. They are stand-ins for visibility, not a test with human readers.
#include "image.hpp"

namespace xe::metrics {

double psnr(const Image& a, const Image& b);
double ssim(const Image& a, const Image& b);

// Binary masks are images holding 0 or 1.
Image dilate(const Image& mask, int radius);
Image erode(const Image& mask, int radius);
Image boundary(const Image& mask, int radius);

// Noise still present: root-mean-square difference between the filtered noisy image and the
// filtered clean image, inside the region.
double remainingNoise(const Image& filteredNoisy, const Image& filteredClean, const Image& region);

// Edge strength kept: gradient of the processed image measured along the clean image's gradient
// direction at the edge pixels, divided by the clean gradient strength. Random noise averages out,
// so a value below 1 means blurred edges and above 1 means sharpened edges.
double edgeRetention(const Image& processed, const Image& clean, const Image& edgePixels);

struct BoundaryStats {
    double contrast;  // mean brightness step across the outline, in display units (what a viewer sees)
    double cnr;       // that step divided by the local spread, which holds both noise and nearby anatomy
};

// Compares the pixels just inside and just outside a structure's outline, tile by tile.
BoundaryStats boundaryStats(const Image& image, const Image& mask, int band = 4, int tile = 24, int minPixels = 12);

// Share of the region pushed to pure black or pure white (detail that can no longer be seen).
double clippedFraction(const Image& image, const Image& region);

}  // namespace xe::metrics
