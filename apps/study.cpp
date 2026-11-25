// Runs every method and setting on every projection and writes one CSV row per result.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "enhance_gpu.hpp"
#include "gl.hpp"
#include "metrics.hpp"

using namespace xe;

namespace {

struct Case {
    std::string id, view, region, split, image, bone, body;
    float scale = 1.0f;  // line integral that corresponds to display value 1
};

struct Method {
    std::string family, name, params;
    std::function<Image(const gpu::Enhancer&, const Image&)> apply;
};

std::vector<Case> readCases(const std::string& folder) {
    std::ifstream in(folder + "/cases.csv");
    if (!in) throw std::runtime_error("cannot open " + folder + "/cases.csv");
    std::string line;
    std::getline(in, line);
    if (line != "id,view,region,split,scale,image,bone,body") throw std::runtime_error("unexpected cases.csv header");
    std::vector<Case> cases;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream row(line);
        Case c;
        std::string scale;
        std::getline(row, c.id, ',');
        std::getline(row, c.view, ',');
        std::getline(row, c.region, ',');
        std::getline(row, c.split, ',');
        std::getline(row, scale, ',');
        std::getline(row, c.image, ',');
        std::getline(row, c.bone, ',');
        std::getline(row, c.body, ',');
        c.scale = std::stof(scale);
        cases.push_back(c);
    }
    return cases;
}

// X-ray quantum noise: photons that get through are Poisson distributed; the log turns counts back
// into a line integral, so thicker body parts come out noisier, as in a real radiograph.
Image addPhotonNoise(const Image& clean, float scale, double photons, unsigned seed) {
    std::mt19937 rng(seed);
    Image noisy(clean.width, clean.height);
    for (size_t i = 0; i < clean.size(); ++i) {
        const double expected = photons * std::exp(-static_cast<double>(clean.pixels[i]) * scale);
        std::poisson_distribution<long long> draw(std::max(expected, 1e-9));
        const double counted = std::max<double>(static_cast<double>(draw(rng)), 1.0);
        noisy.pixels[i] = static_cast<float>(std::clamp(-std::log(counted / photons) / scale, 0.0, 1.0));
    }
    return noisy;
}

std::string text(float v) {
    std::ostringstream s;
    s << v;
    return s.str();
}

std::vector<Method> denoiseMethods() {
    std::vector<Method> methods;
    methods.push_back({"denoise", "none", "-", [](const gpu::Enhancer&, const Image& in) { return in; }});
    for (float sigma : {0.6f, 0.9f, 1.2f, 1.6f})
        methods.push_back({"denoise", "gaussian", "sigma=" + text(sigma),
                           [sigma](const gpu::Enhancer& g, const Image& in) { return g.gaussian(in, sigma); }});
    for (float space : {1.0f, 2.0f, 3.0f})
        for (float range : {0.01f, 0.02f, 0.04f, 0.08f, 0.16f, 0.32f})
            methods.push_back({"denoise", "bilateral", "space=" + text(space) + ";range=" + text(range),
                               [space, range](const gpu::Enhancer& g, const Image& in) {
                                   return g.bilateral(in, space, range);
                               }});
    for (float amount : {0.5f, 1.0f, 2.0f}) {
        methods.push_back({"sharpen", "unsharp", "amount=" + text(amount),
                           [amount](const gpu::Enhancer& g, const Image& in) { return g.unsharp(in, 1.5f, amount); }});
        methods.push_back({"sharpen", "bilateral+unsharp", "amount=" + text(amount),
                           [amount](const gpu::Enhancer& g, const Image& in) {
                               return g.unsharp(g.bilateral(in, 2.0f, 0.02f), 1.5f, amount);
                           }});
    }
    return methods;
}

std::vector<Method> contrastMethods() {
    std::vector<Method> methods;
    methods.push_back({"contrast", "none", "-", [](const gpu::Enhancer&, const Image& in) { return in; }});
    for (float low : {0.5f, 2.0f, 5.0f})
        methods.push_back({"contrast", "auto-window", "percent=" + text(low),
                           [low](const gpu::Enhancer& g, const Image& in) {
                               return g.windowLevel(in, g.autoWindow(in, low, 100.0f - low));
                           }});
    methods.push_back({"contrast", "auto-sigmoid", "percent=2",
                       [](const gpu::Enhancer& g, const Image& in) {
                           return g.windowLevel(in, g.autoWindow(in, 2.0f, 98.0f), cpu::Curve::Sigmoid);
                       }});
    methods.push_back({"contrast", "equalise", "-",
                       [](const gpu::Enhancer& g, const Image& in) { return g.equalise(in); }});
    for (float clip : {1.5f, 2.0f, 3.0f, 4.0f})
        methods.push_back({"contrast", "clahe", "clip=" + text(clip),
                           [clip](const gpu::Enhancer& g, const Image& in) { return g.clahe(in, 8, clip); }});
    for (float gain : {2.0f, 3.0f})
        methods.push_back({"contrast", "local-contrast", "gain=" + text(gain),
                           [gain](const gpu::Enhancer& g, const Image& in) {
                               return g.localContrast(in, 12, 0.5f, gain);
                           }});
    methods.push_back({"contrast", "bilateral+clahe", "clip=2",
                       [](const gpu::Enhancer& g, const Image& in) { return g.clahe(g.bilateral(in, 2.0f, 0.02f), 8, 2.0f); }});
    return methods;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fputs("usage: study <inputs folder> <output folder> [example case id]\n", stderr);
        return 2;
    }
    try {
        const std::string inputs = argv[1], outputs = argv[2];
        const std::string example = argc > 3 ? argv[3] : "";
        const std::vector<Case> cases = readCases(inputs);
        const HeadlessContext context;
        const gpu::Enhancer gpu;
        std::ofstream csv(outputs + "/results.csv");
        if (!csv) throw std::runtime_error("cannot write results.csv");
        csv << "id,view,region,split,photons,family,method,params,psnr,ssim,remaining_noise,edge_retention,"
               "bone_contrast,bone_cnr,clipped_fraction\n";
        std::ofstream info(outputs + "/run_info.txt");
        info << "renderer: " << context.renderer() << "\nversion: " << context.version() << "\ncases: " << cases.size()
             << "\n";

        const std::vector<double> photonLevels = {1e5, 1e4, 2e3};
        std::vector<Method> methods = denoiseMethods();
        const std::vector<Method> contrast = contrastMethods();
        methods.insert(methods.end(), contrast.begin(), contrast.end());

        for (size_t index = 0; index < cases.size(); ++index) {
            const Case& c = cases[index];
            const Image clean = readPfm(inputs + "/" + c.image);
            const Image bone = readPfm(inputs + "/" + c.bone);
            const Image body = readPfm(inputs + "/" + c.body);
            const Image edges = metrics::boundary(bone, 2);
            std::vector<Image> noisy;
            for (size_t level = 0; level < photonLevels.size(); ++level)
                noisy.push_back(addPhotonNoise(clean, c.scale, photonLevels[level],
                                               static_cast<unsigned>(1000 * index + level)));
            for (const Method& m : methods) {
                const Image filteredClean = m.apply(gpu, clean);
                for (size_t level = 0; level < photonLevels.size(); ++level) {
                    const Image result = m.apply(gpu, noisy[level]);
                    const metrics::BoundaryStats outline = metrics::boundaryStats(result, bone);
                    csv << c.id << ',' << c.view << ',' << c.region << ',' << c.split << ','
                        << static_cast<long long>(photonLevels[level]) << ',' << m.family << ',' << m.name << ','
                        << m.params << ',' << metrics::psnr(result, clean) << ',' << metrics::ssim(result, clean) << ','
                        << metrics::remainingNoise(result, filteredClean, body) << ','
                        << metrics::edgeRetention(result, clean, edges) << ',' << outline.contrast << ',' << outline.cnr
                        << ',' << metrics::clippedFraction(result, body) << '\n';
                    const bool keep = c.id + "_" + c.view == example && level == 1;
                    if (keep) writePfm(outputs + "/example_" + m.name + "_" + m.params + ".pfm", result);
                }
            }
            if (c.id + "_" + c.view == example) {
                writePfm(outputs + "/example_clean.pfm", clean);
                writePfm(outputs + "/example_noisy.pfm", noisy[1]);
            }
            std::printf("case %zu/%zu %s %s done\n", index + 1, cases.size(), c.id.c_str(), c.view.c_str());
            std::fflush(stdout);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
    return 0;
}
