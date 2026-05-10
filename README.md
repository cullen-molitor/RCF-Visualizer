# RCF-Visualizer

An interactive R Shiny app for exploring the MOSAIKS random convolutional features (RCF) featurization pipeline described in [Rolf et al. 2021, *Nature Communications*](https://doi.org/10.1038/s41467-021-24638-z).

## What it does

For each of three user-selected (or random) locations on an image, the app runs the full MOSAIKS featurization pipeline and visualizes each step:

1. **Extract** — pull an M×M×3 raw patch centered at the selected location
2. **Whiten** — ZCA whitening of patch to decorelate the pixels 
3. **Convolve** — slide the whitened patch as a kernel across the full image
4. **Activate** — apply bias and ReLU: `A_k = ReLU(W_k * I + b)`
5. **Pool** — globally average the activation map to produce scalar feature `x_k(I)`

The three selections are color-coded (magenta / blue / green), and results are displayed side-by-side.

## Usage

[Try online!](https://mosaiks.shinyapps.io/rcf-visualizer/)

Or run locally by cloning this repository. Additional images (PNG or JPEG) can be added to the `images/` directory. A handful of example images are included.

## Controls

| Control | Description |
|---|---|
| **Image** | Select from images in the `images/` directory |
| **Patch selection** | Click three points on the image, or generate them from a random seed |
| **Kernel size (M)** | Patch/kernel side length in pixels (3–9) |
| **Bias (b)** | Scalar added to conv output before ReLU; more negative suppresses more of the map |
| **Heatmap palette** | Color scale for convolution and activation maps |
| **Compute resolution** | Long-edge cap for the image used during computation (128–512 px); higher = more spatial detail but slower |

## Reference

Rolf, E., et al. (2021). A generalizable and accessible approach to machine learning with global satellite imagery. *Nature Communications*, 12, 4392. https://doi.org/10.1038/s41467-021-24638-z
