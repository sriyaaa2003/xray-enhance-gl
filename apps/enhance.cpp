// Command-line tool: apply one shader operation to a greyscale PFM image.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "enhance_gpu.hpp"
#include "gl.hpp"

namespace {

const char* kUsage =
    "usage: enhance <input.pfm> <output.pfm> <operation> [numbers]\n"
    "  window <level> <width>      sigmoid <level> <width>     auto <low%> <high%>\n"
    "  equalise                    match <reference.pfm>       clahe <tiles> <clipLimit>\n"
    "  gaussian <sigma>            bilateral <sigmaSpace> <sigmaRange>\n"
    "  unsharp <sigma> <amount>    local <radius> <strength> <maxGain>\n";

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    try {
        const xe::Image input = xe::readPfm(argv[1]);
        const std::string operation = argv[3];
        auto number = [&](int i) {
            if (4 + i >= argc) throw std::invalid_argument("missing number for " + operation);
            return std::stof(argv[4 + i]);
        };
        const xe::HeadlessContext context;
        const xe::gpu::Enhancer gpu;
        xe::Image output;
        if (operation == "window") output = gpu.windowLevel(input, {number(0), number(1)});
        else if (operation == "sigmoid") output = gpu.windowLevel(input, {number(0), number(1)}, xe::cpu::Curve::Sigmoid);
        else if (operation == "auto") output = gpu.windowLevel(input, gpu.autoWindow(input, number(0), number(1)));
        else if (operation == "equalise") output = gpu.equalise(input);
        else if (operation == "match") {
            if (argc < 5) throw std::invalid_argument("match needs a reference image");
            output = gpu.matchHistogram(input, xe::readPfm(argv[4]));
        }
        else if (operation == "clahe") output = gpu.clahe(input, static_cast<int>(number(0)), number(1));
        else if (operation == "gaussian") output = gpu.gaussian(input, number(0));
        else if (operation == "bilateral") output = gpu.bilateral(input, number(0), number(1));
        else if (operation == "unsharp") output = gpu.unsharp(input, number(0), number(1));
        else if (operation == "local") output = gpu.localContrast(input, static_cast<int>(number(0)), number(1), number(2));
        else throw std::invalid_argument("unknown operation: " + operation);
        xe::writePfm(argv[2], output);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n%s", error.what(), kUsage);
        return 1;
    }
    return 0;
}
