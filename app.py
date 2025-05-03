import streamlit as st
from PIL import Image, UnidentifiedImageError
import numpy as np
import tempfile
import os
import sys

from artutils.io_utils import load_and_resize_image
from artutils.clustering import fit_kmeans, fit_gmm, gmm_soft_gradient
from artutils.color_utils import get_hex_codes_from_centers, get_hex_codes_from_gmm_means, generate_full_hsl_gradient, generate_opposite_palette

from artutils.palette_tools import extract_palette_by_frequency_and_lab
from artutils.visualization import plot_swatch, plot_wheel


def save_uploaded_file(uploaded_file):
    """Save an upload to a temp file and return that path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ———————————————————————————————
#  STREAMLIT CONFIG
# ———————————————————————————————
st.set_page_config(page_title="ArtUtils Palette Studio", layout="centered")
st.title("🎨 ArtUtils Palette Studio")
st.write(
    """
    **artutils** helps you extract and explore colors from any image—  
    either a piece of artwork or a palette graphic.  
    Choose a method below, generate your palette or gradient, and download the results.
    """
)


# ———————————————————————————————
#  STEP 1 – UPLOAD
# ———————————————————————————————

upload_type = st.radio("🧩 What are you uploading?", ["Artwork Image", "Palette Image"])
st.info("Upload a photograph, painting, screenshot, or any image file. If you already have a palette (e.g. a PNG of color swatches), choose *Palette Image*.")
uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
try:
    img = Image.open(uploaded)
    img.verify()         # will throw if not a real image
except UnidentifiedImageError:
    st.error("That file doesn’t look like a valid image.")
    st.stop()
if not uploaded:
    st.stop()

tmp_path = save_uploaded_file(uploaded)
st.image(tmp_path, caption="Uploaded", use_container_width=True)

try:    
    # ———————————————————————————————
    #  STEP 2 – ARTWORK MODES
    # ———————————————————————————————
    if upload_type == "Artwork Image":
    
        st.subheader("Select what you want to extract:")
        method = st.radio(
            "",
            ["Hard Palette", "Soft Palette", "Moody Gradient"]
        )
        with st.expander("What do these modes mean?"):
            st.markdown(
                """
                - **Hard Palette (KMeans)**  
                  Finds dominant colors by partitioning pixels into clusters.  
                - **Soft Palette (GMM-based)**  
                  Uses soft clustering that overlaps to find color centers; each pixel can belong to multiple centers, giving a richer set of tones.  
                - **Moody Gradient**  
                  Blends soft clustering into a continuous color wheel that loops seamlessly around the hue spectrum.
                """
            )
    
        auto_colors = st.checkbox(
            "Let the model automatically choose the number of colors",
            value=True
        )
        if auto_colors:
            st.caption("We’ll pick a sensible default so you don’t need to guess.🧸")
        else:
            num_colors = st.slider(
                "Number of colors",
                2, 20, 6,
                help="Adjust how many distinct colors you want to extract. 🧲"
            )
    
        if st.button("Generate"):
    
            img = load_and_resize_image(tmp_path, size=(100, 100))
            pixels = img.reshape(-1, 3)
    
            # — Hard Palette —
            if method == "Hard Palette":
                k = num_colors if not auto_colors else None
                km = fit_kmeans(pixels, max_k=(k or 8), use_elbow=auto_colors)
                palette = get_hex_codes_from_centers(km.cluster_centers_)
    
                swatch_path = "hard_palette.png"
                plot_swatch(palette, save_path=swatch_path)
                st.subheader("Hard Palette")
                st.image(swatch_path)
                st.code(palette, language="bash")
    
                st.download_button(
                    "Download Swatch PNG",
                    open(swatch_path, "rb").read(),
                    file_name=swatch_path,
                    mime="image/png"
                )
                st.download_button(
                    "Download HEX Codes",
                    "\n".join(palette),
                    file_name="hard_palette.txt",
                    mime="text/plain"
                )
    
            # — Soft Palette —
            elif method == "Soft Palette":
                k = num_colors if not auto_colors else None
                gmm = fit_gmm(pixels, max_k=(k or 6), use_bic=auto_colors)
                palette = get_hex_codes_from_gmm_means(gmm.means_)
    
                swatch_path = "soft_palette.png"
                plot_swatch(palette, save_path=swatch_path)
                st.subheader("Soft Palette (GMM Centers)")
                st.image(swatch_path)
                st.code(palette, language="bash")
    
                st.download_button(
                    "Download Swatch PNG",
                    open(swatch_path, "rb").read(),
                    file_name=swatch_path,
                    mime="image/png"
                )
                st.download_button(
                    "Download HEX Codes",
                    "\n".join(palette),
                    file_name="soft_palette.txt",
                    mime="text/plain"
                )
    
            # — Moody Gradient —
            else:
                k = num_colors if not auto_colors else 6
                palette = gmm_soft_gradient(tmp_path, n_components=k)
    
                wheel_path = "moody_gradient.png"
                plot_wheel(palette, save_path=wheel_path)
                st.subheader("Moody Gradient (Soft GMM)")
                st.image(wheel_path)
                st.code(palette, language="bash")
    
                st.download_button(
                    "Download Gradient PNG",
                    open(wheel_path, "rb").read(),
                    file_name=wheel_path,
                    mime="image/png"
                )
                st.download_button(
                    "Download HEX Codes",
                    "\n".join(palette),
                    file_name="moody_gradient.txt",
                    mime="text/plain"
                )
    
    
    # — Palette Image branch —
    else:
    
        st.subheader("Extracted Palette")
        st.info("We look at color frequency and their differences to give you a clean set of swatches.")
        palette = extract_palette_by_frequency_and_lab(tmp_path)
    
        swatch_path = "extracted_palette.png"
        plot_swatch(palette, save_path=swatch_path)
        st.image(swatch_path)
        st.code(palette, language="bash")
    
        st.download_button(
            "Download HEX Codes",
            "\n".join(palette),
            file_name="extracted_palette.txt",
            mime="text/plain"
        )
    
        st.subheader("📍 Transform Palette")
        st.markdown(
            "Choose how to work with your extracted swatches:\n"
            "- **Generate Gradient**: smooth transitions between each pair of colors\n"
            "- **Generate Opposites**: flip each color to get complementary colors"
        )
        task = st.selectbox("", ["Generate Gradient", "Generate Opposites"])
    
        if task == "Generate Gradient":
            steps = st.slider(
                "Steps per transition",
                10, 100, 50,
                help="More steps = smoother color wheel"
            )
            grad = generate_full_hsl_gradient(palette, steps_per_transition=steps)
    
            wheel_path = "hsl_gradient.png"
            plot_wheel(grad, save_path=wheel_path)
            st.subheader("HSL Gradient Wheel")
            st.image(wheel_path)
            st.code(grad, language="bash")
    
            st.download_button(
                "Download Gradient PNG",
                open(wheel_path, "rb").read(),
                file_name=wheel_path,
                mime="image/png"
            )
            st.download_button(
                "Download HEX Codes",
                "\n".join(grad),
                file_name="hsl_gradient.txt",
                mime="text/plain"
            )
    
        else:
            opposites = generate_opposite_palette(palette)
    
            wheel_path = "opposite_wheel.png"
            plot_wheel(opposites, save_path=wheel_path)
            st.subheader("Opposite Color Wheel")
            st.image(wheel_path)
            st.code(opposites, language="bash")
    
            st.download_button(
                "Download Opposites PNG",
                open(wheel_path, "rb").read(),
                file_name=wheel_path,
                mime="image/png"
            )
            st.download_button(
                "Download HEX Codes",
                "\n".join(opposites),
                file_name="opposite_palette.txt",
                mime="text/plain"
            )
    
    
    # — CLEANUP —
finally:
    os.remove(tmp_path)
