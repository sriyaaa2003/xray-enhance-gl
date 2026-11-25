#include "image.hpp"

#include <fstream>
#include <stdexcept>

namespace xe {

Image readPfm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string magic;
    int w = 0, h = 0;
    float scale = 0.0f;
    in >> magic >> w >> h >> scale;
    if (magic != "Pf" || w <= 0 || h <= 0 || scale >= 0.0f)
        throw std::runtime_error("expected a little-endian greyscale PFM: " + path);
    in.get();  // the single whitespace byte that ends the header
    Image image(w, h);
    for (int y = h - 1; y >= 0; --y)
        in.read(reinterpret_cast<char*>(&image.at(0, y)), static_cast<std::streamsize>(w * sizeof(float)));
    if (!in) throw std::runtime_error("file is shorter than its header says: " + path);
    return image;
}

void writePfm(const std::string& path, const Image& image) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "Pf\n" << image.width << ' ' << image.height << "\n-1.0\n";
    for (int y = image.height - 1; y >= 0; --y)
        out.write(reinterpret_cast<const char*>(&image.pixels[static_cast<size_t>(y) * image.width]),
                  static_cast<std::streamsize>(image.width * sizeof(float)));
}

}  // namespace xe
