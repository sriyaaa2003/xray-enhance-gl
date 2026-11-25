#version 430 core
// Adaptive local contrast: stretch detail around the local mean, more strongly where the
// neighbourhood is flat, with a ceiling on the gain so that noise is not blown up without limit.
layout(binding = 0) uniform sampler2D uImage;
uniform int uRadius;
uniform float uStrength;
uniform float uMaxGain;
uniform float uGlobalDeviation;
layout(location = 0) out float outValue;

void main() {
    ivec2 last = textureSize(uImage, 0) - 1;
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float centre = texelFetch(uImage, pixel, 0).r;
    // Sums of differences from the centre pixel keep 32-bit precision usable.
    float sum = 0.0;
    float squares = 0.0;
    for (int dy = -uRadius; dy <= uRadius; ++dy) {
        for (int dx = -uRadius; dx <= uRadius; ++dx) {
            float d = texelFetch(uImage, clamp(pixel + ivec2(dx, dy), ivec2(0), last), 0).r - centre;
            sum += d;
            squares += d * d;
        }
    }
    float count = float((2 * uRadius + 1) * (2 * uRadius + 1));
    float meanDifference = sum / count;
    float mean = centre + meanDifference;
    float deviation = sqrt(max(squares / count - meanDifference * meanDifference, 0.0));
    float gain = min(max(uStrength * uGlobalDeviation / (deviation + 1e-4), 1.0), uMaxGain);
    outValue = clamp(mean + gain * (centre - mean), 0.0, 1.0);
}
