#version 430 core
// CLAHE, second stage: each tile has its own table; blend the four nearest ones so tile edges do not show.
layout(binding = 0) uniform sampler2D uImage;
layout(binding = 1) uniform sampler2D uTables;  // width = bins, height = number of tiles
uniform ivec2 uTiles;
layout(location = 0) out float outValue;

void main() {
    ivec2 size = textureSize(uImage, 0);
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    int bins = textureSize(uTables, 0).x;
    float value = clamp(texelFetch(uImage, pixel, 0).r, 0.0, 1.0);
    int bin = int(value * float(bins - 1) + 0.5);

    // Position in tile units, with tile centres on whole numbers.
    vec2 tileSize = vec2(size) / vec2(uTiles);
    vec2 grid = (vec2(pixel) + 0.5) / tileSize - 0.5;
    ivec2 base = ivec2(floor(grid));
    vec2 f = grid - vec2(base);
    ivec2 low = clamp(base, ivec2(0), uTiles - 1);
    ivec2 high = clamp(base + 1, ivec2(0), uTiles - 1);

    float a = texelFetch(uTables, ivec2(bin, low.y * uTiles.x + low.x), 0).r;
    float b = texelFetch(uTables, ivec2(bin, low.y * uTiles.x + high.x), 0).r;
    float c = texelFetch(uTables, ivec2(bin, high.y * uTiles.x + low.x), 0).r;
    float d = texelFetch(uTables, ivec2(bin, high.y * uTiles.x + high.x), 0).r;
    float top = a * (1.0 - f.x) + b * f.x;
    float bottom = c * (1.0 - f.x) + d * f.x;
    outValue = top * (1.0 - f.y) + bottom * f.y;
}
