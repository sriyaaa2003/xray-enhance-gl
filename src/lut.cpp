#include "lut.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace xe {

namespace {

std::vector<double> cumulative(const std::vector<uint32_t>& histogram) {
    std::vector<double> cdf(histogram.size());
    double running = 0.0;
    for (size_t i = 0; i < histogram.size(); ++i) {
        running += histogram[i];
        cdf[i] = running;
    }
    return cdf;
}

}  // namespace

Window percentileWindow(const std::vector<uint32_t>& histogram, float lowPercent, float highPercent) {
    if (histogram.size() < 2 || !(lowPercent < highPercent)) throw std::invalid_argument("bad percentile window");
    const std::vector<double> cdf = cumulative(histogram);
    const double total = cdf.back();
    const int last = static_cast<int>(histogram.size()) - 1;
    auto binAt = [&](double fraction) {
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), std::max(fraction * total, 1.0));
        return std::min(static_cast<int>(it - cdf.begin()), last);
    };
    const float low = static_cast<float>(binAt(lowPercent / 100.0)) / last;
    const float high = static_cast<float>(binAt(highPercent / 100.0)) / last;
    return {0.5f * (low + high), std::max(high - low, 1.0f / last)};
}

std::vector<float> equalisationLut(const std::vector<uint32_t>& histogram) {
    const std::vector<double> cdf = cumulative(histogram);
    const int last = static_cast<int>(histogram.size()) - 1;
    const double total = cdf.back();
    double first = 0.0;
    for (double value : cdf)
        if (value > 0.0) {
            first = value;
            break;
        }
    std::vector<float> lut(histogram.size());
    for (size_t i = 0; i < lut.size(); ++i) {
        if (total - first <= 0.0)
            lut[i] = static_cast<float>(i) / last;  // flat image: leave it unchanged
        else
            lut[i] = static_cast<float>(std::clamp((cdf[i] - first) / (total - first), 0.0, 1.0));
    }
    return lut;
}

std::vector<float> matchingLut(const std::vector<uint32_t>& source, const std::vector<uint32_t>& reference) {
    if (source.size() != reference.size()) throw std::invalid_argument("histograms differ in size");
    const std::vector<double> cdfSource = cumulative(source);
    const std::vector<double> cdfReference = cumulative(reference);
    const double totalSource = cdfSource.back();
    const double totalReference = cdfReference.back();
    const int last = static_cast<int>(source.size()) - 1;
    std::vector<float> lut(source.size());
    int j = 0;  // both distributions only grow, so one forward walk is enough
    for (size_t i = 0; i < source.size(); ++i) {
        const double wanted = cdfSource[i] / totalSource;
        while (j < last && cdfReference[j] / totalReference < wanted - 1e-12) ++j;
        lut[i] = static_cast<float>(j) / last;
    }
    return lut;
}

std::vector<float> claheLuts(const std::vector<uint32_t>& tileHistograms, int bins, int tileCount, float clipLimit) {
    if (static_cast<size_t>(bins) * tileCount != tileHistograms.size())
        throw std::invalid_argument("tile histograms have the wrong size");
    std::vector<float> luts(tileHistograms.size());
    std::vector<double> clipped(bins);
    for (int tile = 0; tile < tileCount; ++tile) {
        const size_t offset = static_cast<size_t>(tile) * bins;
        double total = 0.0;
        for (int b = 0; b < bins; ++b) total += tileHistograms[offset + b];
        const double limit = std::max(1.0, static_cast<double>(clipLimit) * total / bins);
        double excess = 0.0;
        for (int b = 0; b < bins; ++b) {
            const double count = tileHistograms[offset + b];
            clipped[b] = std::min(count, limit);
            excess += count - clipped[b];
        }
        double running = 0.0;
        for (int b = 0; b < bins; ++b) {
            running += clipped[b] + excess / bins;
            luts[offset + b] = total > 0.0 ? static_cast<float>(std::min(running / total, 1.0))
                                           : static_cast<float>(b) / (bins - 1);
        }
    }
    return luts;
}

}  // namespace xe
