# artutils

A modular Python library for extracting, generating, and visualizing color palettes directly from images. Created for digital artists, illustrators, designers, and anyone who wants to use image-driven color workflows with perceptual intelligence and ML-assisted tools.

---

## What Is It?

**artutils** helps you:
- Extract clean, meaningful color palettes from real images
- Generate full-spectrum gradients and complementary color schemes
- Visualize your color designs as swatches or wheels
- Create smooth blended palettes using Gaussian Mixture Models

---

## What's Inside (And Why It Matters)

### Palette Extraction (`palette_tools.py`)
**Why:** Picking random pixels creates duplicates and noise. This tool:
- Uses color frequency
- Converts to perceptual LAB space
- Deduplicates based on Delta E (color difference)

**Use it for:**
- Building consistent palettes from reference art or moodboards  
- Feeding a painting layer or UI theme system  

---

### Clustering with KMeans / GMM (`clustering.py`)
**Why:** Hard clustering (KMeans) finds dominant color regions, while GMM captures **soft overlaps**.

**Use it for:**
- Logo and brand palette design  
- Color structure discovery from photo datasets  
- Creating refined palettes from photography

---

### Soft GMM Gradient (`gmm_soft_gradient`)
**Why:** Real-world transitions are fuzzy. This tool:
- Assigns pixels soft probabilities across color centers
- Blends colors based on those weights
- Produces smooth, loopable color wheels

**Use it for:**
- Background gradient generation  
- Data viz color transitions  
- Mood palette exploration for art

---

### Color Utilities (`color_utils.py`)
Includes:
- hex ↔ RGB ↔ LAB converters  
- HSL gradient interpolation  
- Opposite color generator  

**Use it for:**
- Theme toggling  
- Exploring lightness/saturation transitions  
- Creating color harmonies and contrasts

---

### Visualization (`visualization.py`)
- Swatch row plots  
- Circular polar color wheels  
- Optional export to `.png`

**Use it for:**
- Portfolio graphics  
- Color reviews with clients or teammates  
- Visual debugging of generated palettes

---
## Project Structure
```
artutils/
├── __init__.py
├── io_utils.py
├── color_utils.py
├── clustering.py
├── palette_tools.py
├── visualization.py
└── examples/
    ├── clustering_example.py
    ├── palette_example.py
    ├── gmm_gradient_example.py
    └── visualization_example.py
```

---
## Coming Sometime in the Future: Soft Brush Texture Tools

I was actively prototyping a system to extract high-resolution brush textures from source images using:

- Local contrast enhancement (e.g., CLAHE)
- Texture descriptors like LBP + entropy
- Deduplication of visually similar tiles
- KMeans clustering to create grouped brush packs

**Goal:** help digital artists generate reusable brush alphas from skin, fabric, watercolor, etc.  
This feature is paused for now but is planned for **v2**.

---
## Recommended: Use a virtual environment

To avoid dependency conflicts, we recommend using a virtual environment:

<pre>
python -m venv venv_
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install artutils

</pre>
## Getting Started

Clone and install the library:
<pre>
git clone https://github.com/aki-au/artutils.git
cd artutils
pip install -e .
</pre>

---
If you’d like to collaborate or feature this project somewhere, feel free to reach out!


## PyPi Modular Package
https://pypi.org/project/artutils/1.0.0/ --> This is where you can download it through PyPi!

## Medium Article (with Pictures and Stuff)

