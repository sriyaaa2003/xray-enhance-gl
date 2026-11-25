#include "enhance_gpu.hpp"

namespace xe::gpu {

Enhancer::Enhancer()
    : windowLevel_(Program::fragment("window_level.frag")),
      histogram_(Program::compute("histogram.comp")),
      lut_(Program::fragment("lut.frag")),
      clahe_(Program::fragment("clahe.frag")),
      gaussian_(Program::fragment("gaussian.frag")),
      bilateral_(Program::fragment("bilateral.frag")),
      unsharp_(Program::fragment("unsharp.frag")),
      localContrast_(Program::fragment("local_contrast.frag")) {}

Image Enhancer::windowLevel(const Image& in, Window window, Curve curve) const {
    const Texture source(in);
    const Texture target(in.width, in.height);
    source.bind(0);
    windowLevel_.set("uLevel", window.level);
    windowLevel_.set("uWidth", window.width);
    windowLevel_.set("uCurve", static_cast<int>(curve));
    pass_.run(windowLevel_, target);
    return target.read();
}

std::vector<uint32_t> Enhancer::histogram(const Image& in, int bins, int tilesX, int tilesY) const {
    const Texture source(in);
    std::vector<uint32_t> counts(static_cast<size_t>(bins) * tilesX * tilesY, 0u);
    const GLsizeiptr bytes = static_cast<GLsizeiptr>(counts.size() * sizeof(uint32_t));
    GLuint buffer = 0;
    glGenBuffers(1, &buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, counts.data(), GL_DYNAMIC_READ);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
    source.bind(0);
    histogram_.set("uBins", bins);
    histogram_.set("uTiles", tilesX, tilesY);
    glUseProgram(histogram_.id());
    glDispatchCompute(static_cast<GLuint>((in.width + 7) / 8), static_cast<GLuint>((in.height + 7) / 8), 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes, counts.data());
    glDeleteBuffers(1, &buffer);
    checkGl("histogram");
    return counts;
}

Window Enhancer::autoWindow(const Image& in, float lowPercent, float highPercent) const {
    return percentileWindow(histogram(in, 4096), lowPercent, highPercent);
}

Image Enhancer::applyLut(const Image& in, const std::vector<float>& lut) const {
    const Texture source(in);
    const Texture table(static_cast<int>(lut.size()), 1, lut.data());
    const Texture target(in.width, in.height);
    source.bind(0);
    table.bind(1);
    pass_.run(lut_, target);
    return target.read();
}

Image Enhancer::equalise(const Image& in, int bins) const {
    return applyLut(in, equalisationLut(histogram(in, bins)));
}

Image Enhancer::matchHistogram(const Image& in, const Image& reference, int bins) const {
    return applyLut(in, matchingLut(histogram(in, bins), histogram(reference, bins)));
}

Image Enhancer::clahe(const Image& in, int tiles, float clipLimit, int bins) const {
    const std::vector<float> luts = claheLuts(histogram(in, bins, tiles, tiles), bins, tiles * tiles, clipLimit);
    const Texture source(in);
    const Texture tables(bins, tiles * tiles, luts.data());
    const Texture target(in.width, in.height);
    source.bind(0);
    tables.bind(1);
    clahe_.set("uTiles", tiles, tiles);
    pass_.run(clahe_, target);
    return target.read();
}

Image Enhancer::gaussian(const Image& in, float sigma) const {
    const Texture source(in);
    const Texture middle(in.width, in.height);
    const Texture target(in.width, in.height);
    gaussian_.set("uSigma", sigma);
    gaussian_.set("uRadius", xe::cpu::gaussianRadius(sigma));
    source.bind(0);
    gaussian_.set("uDirection", 1, 0);
    pass_.run(gaussian_, middle);
    middle.bind(0);
    gaussian_.set("uDirection", 0, 1);
    pass_.run(gaussian_, target);
    return target.read();
}

Image Enhancer::bilateral(const Image& in, float sigmaSpace, float sigmaRange) const {
    const Texture source(in);
    const Texture target(in.width, in.height);
    source.bind(0);
    bilateral_.set("uRadius", xe::cpu::bilateralRadius(sigmaSpace));
    bilateral_.set("uSigmaSpace", sigmaSpace);
    bilateral_.set("uSigmaRange", sigmaRange);
    pass_.run(bilateral_, target);
    return target.read();
}

Image Enhancer::unsharp(const Image& in, float sigma, float amount) const {
    const Image blurred = gaussian(in, sigma);
    const Texture source(in);
    const Texture low(blurred);
    const Texture target(in.width, in.height);
    source.bind(0);
    low.bind(1);
    unsharp_.set("uAmount", amount);
    pass_.run(unsharp_, target);
    return target.read();
}

Image Enhancer::localContrast(const Image& in, int radius, float strength, float maxGain) const {
    const Texture source(in);
    const Texture target(in.width, in.height);
    source.bind(0);
    localContrast_.set("uRadius", radius);
    localContrast_.set("uStrength", strength);
    localContrast_.set("uMaxGain", maxGain);
    localContrast_.set("uGlobalDeviation", xe::cpu::standardDeviation(in));
    pass_.run(localContrast_, target);
    return target.read();
}

}  // namespace xe::gpu
