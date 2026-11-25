#version 430 core
// Replaces every intensity using a look-up table (histogram equalisation and histogram matching).
layout(binding = 0) uniform sampler2D uImage;
layout(binding = 1) uniform sampler2D uTable;  // width = number of bins, height = 1
layout(location = 0) out float outValue;

void main() {
    float value = clamp(texelFetch(uImage, ivec2(gl_FragCoord.xy), 0).r, 0.0, 1.0);
    int bins = textureSize(uTable, 0).x;
    int bin = int(value * float(bins - 1) + 0.5);
    outValue = texelFetch(uTable, ivec2(bin, 0), 0).r;
}
