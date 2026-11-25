#pragma once
// Host-side steps shared by the GPU and CPU paths: turning histograms into look-up tables.
#include <cstdint>
#include <vector>

namespace xe {

struct Window {
    float level;  // intensity shown as middle grey
    float width;  // intensity range spread over black to white
};

// Level and width that cover the given percentiles of the histogram (intensities in [0, 1]).
Window percentileWindow(const std::vector<uint32_t>& histogram, float lowPercent, float highPercent);

// Classic histogram equalisation.
std::vector<float> equalisationLut(const std::vector<uint32_t>& histogram);

// Maps source intensities so that their distribution follows the reference distribution.
std::vector<float> matchingLut(const std::vector<uint32_t>& source, const std::vector<uint32_t>& reference);

// One table per tile for contrast-limited adaptive histogram equalisation (CLAHE).
// clipLimit is a multiple of the mean bin count; counts above it are cut and spread evenly over all bins.
std::vector<float> claheLuts(const std::vector<uint32_t>& tileHistograms, int bins, int tileCount, float clipLimit);

}  // namespace xe
