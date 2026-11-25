# How it works

## 1. The big picture

1. An image is a grid of floating-point numbers between 0 (black) and 1 (white).
2. It is uploaded to the graphics side as a **texture** (`Texture` in `src/gl.cpp`, format `R32F`: one 32-bit float per pixel).
3. A **shader** is a small program that runs once for every pixel, many pixels at the same time.
4. The result is written into a second texture and read back into C++.

A real viewer would keep the result on the graphics card and draw it on screen. This project reads it back
so that it can be tested and measured.

## 2. Running OpenGL without a window (`HeadlessContext`)

OpenGL needs a "context" before any call works. Usually a window library creates it. A build server has no
screen, so the code uses **EGL** with Mesa's *surfaceless* platform. The renderer on the build server is
**llvmpipe**: OpenGL running on the processor. That is why this project reports **no speed numbers**.

## 3. How a fragment shader runs over an image (`FullscreenPass`)

- The target texture is attached to a **framebuffer**, so drawing writes into the texture.
- The vertex shader (`fullscreen.vert`) draws one huge triangle that covers the whole area. It needs no vertex data.
- The fragment shader then runs once per pixel. `gl_FragCoord.xy` says which pixel it is.
- `texelFetch` reads an exact pixel with no blending. `texture()` would interpolate, which we do not want here.
- At the image border, coordinates are clamped, so edge pixels repeat.

**Why fragment shaders for filters?** It is how a viewer filters what it displays, and it works on older
OpenGL too. **Why a compute shader for the histogram?** A fragment shader writes only to its own pixel. A
histogram needs thousands of pixels to add into the same few counters.

## 4. The histogram (`histogram.comp`)

- Work is split into groups of 8 by 8 threads. Threads outside the image return immediately.
- Each thread finds its pixel's bin and calls **`atomicAdd`**. Without "atomic", two threads could read the same
  old count and one addition would be lost.
- `glMemoryBarrier` makes sure all writes have finished before C++ reads the buffer.
- With tiles, each tile gets its own block of counters. This feeds CLAHE.

## 5. Intensity mappings

- **Window/level** (`window_level.frag`): `level` is the intensity shown as middle grey; `width` is the range
  spread from black to white. Values outside are clipped, so their detail is gone from the screen.
- **Sigmoid**: the DICOM soft version. It squeezes the ends smoothly instead of cutting them.
- **Automatic window** (`percentileWindow`): take the histogram and pick, for example, the 2nd and 98th
  percentile as the black and white points. Percentiles ignore a few extreme pixels.

## 6. Histogram methods (`src/lut.cpp`, `lut.frag`, `clahe.frag`)

A **look-up table** says "intensity bin 17 becomes 0.31". The CPU builds the table from the histogram; the
shader applies it.

- **Equalisation**: the table is the running total of the histogram (the CDF), scaled to 0..1. Crowded
  intensity ranges get spread out.
- **Matching**: for each source bin, find the first reference bin whose CDF is at least as large. The image then
  has about the same intensity distribution as the reference.
- **CLAHE**: equalisation per tile, with two fixes.
  - *Clip limit*: a flat area makes one tall histogram spike, and equalising it would blow up noise. Counts above
    the limit are cut off and shared equally among all bins.
  - *Blending*: each pixel mixes the tables of the four nearest tile centres (bilinear), so no tile borders show.

## 7. Filters

- **Gaussian** (`gaussian.frag`): a weighted average where near pixels count more. It is **separable**: one pass
  across, then one down. For radius r that is about 2(2r+1) reads instead of (2r+1) squared.
- **Bilateral** (`bilateral.frag`): like Gaussian, but a neighbour also loses weight if its brightness differs a
  lot. Across a bone edge the difference is large, so the edge is mostly not mixed. It is not separable. A small
  `sigmaRange` keeps edges but removes little noise.
- **Unsharp mask** (`unsharp.frag`): `image + amount x (image - blurred)`. "Image minus blurred" is the fine
  detail, and noise is fine detail too.
- **Adaptive local contrast** (`local_contrast.frag`): push each pixel away from its local mean. The push is
  larger where the neighbourhood is flat and is capped by `maxGain`.

## 8. Testing the shaders

Every method also exists in plain C++ (`src/enhance_cpu.cpp`). The tests run both on the same input.

- Histograms must match **exactly** (they are whole numbers).
- Images must match within 0.0001 on a 0..1 scale. Floating-point results may differ in the last digits.
- Property tests check things that must be true anyway: a flat image stays flat when blurred; a window over the
  full range changes nothing; CLAHE tables never decrease.
- The test image's size is not a multiple of 8, to catch mistakes at the edges.

**What these tests do not prove:** the look-up tables for equalisation, matching and CLAHE are built by the same
code (`src/lut.cpp`) in both paths, so comparing the two cannot catch a mistake there. Also, both versions follow the same formulas written by the same author. They catch
shader bugs, not a wrong formula. The property tests help with that.

## 9. The study (`apps/study.cpp`, `tools/`)

- **Data:** public CT scans (TotalSegmentator small subset). `make_inputs.py` adds up attenuation along one axis
  to make a flat projection. This is a **parallel** projection, simpler than a real X-ray.
- **Noise:** fewer photons means more noise. Photon counts follow a Poisson distribution, then a logarithm turns
  counts back into a line integral.
- **Tuning and unseen subjects:** settings are picked on one third of the subjects and reported on the other two
  thirds. Both views of a subject stay on the same side.
- **Measures**
  - *PSNR, SSIM*: closeness to the noise-free image. Used only for noise removal.
  - *Noise left*: difference between filtering the noisy image and filtering the clean one.
  - *Edge strength kept*: gradient along the clean image's gradient direction at bone outlines. Noise averages
    out, so it measures the real edge.
  - *Bone-edge step on screen*: brightness difference across the outline. This is what the eye gets.
  - *Step divided by local spread* (a contrast-to-noise ratio). The spread includes noise **and** the
    surrounding anatomy. **A window changes the step but hardly changes this ratio**, because it scales
    both together (unless it clips).

None of these is a test with human readers or radiotherapy staff.
