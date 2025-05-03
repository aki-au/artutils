# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tempfile, os

from artutils.io_utils import load_and_resize_image
from artutils.clustering import fit_kmeans, gmm_soft_gradient, fit_gmm
from artutils.color_utils import get_hex_codes_from_centers, generate_full_hsl_gradient, generate_opposite_palette
from artutils.palette_tools import extract_palette_by_frequency_and_lab
from artutils.visualization import plot_swatch, plot_wheel


def save_uploaded_file(uploaded_file):
    """Save a Streamlit-uploaded file to a temp PNG and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ———————————————————————————————
#  STREAMLIT UI SETUP
# ———————————————————————————————
st.set_page_config(page_title="ArtUtils Palette Studio", layout="centered")
st.title("ArtUtils Palette Studio")
st.write("Upload an artwork or a palette image, then choose how to extract or transform colors.")


# ———————————————————————————————
#  STEP 1: UPLOAD
# ———————————————————————————————
upload_type = st.radio(
    "What are you uploading?",
    ["Artwork Image", "Palette Image"]
)

uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
if not uploaded:
    st.stop()


tmp_path = save_uploaded_file(uploaded)
st.image(tmp_path, caption="Uploaded", use_column_width=True)


# ———————————————————————————————
#  STEP 2: OPTIONS & CONTROLS
# ———————————————————————————————
if upload_type == "Artwork Image":
    st.subheader("Select what you want to extract:")
    method = st.radio(
        "",
        ["Hard Palette", "Soft Palette", "Moody Gradient"]
    )

    auto_colors = st.checkbox("Let the model automatically choose number of colors", value=True)
    if not auto_colors:
        num_colors = st.slider("Number of colors", 2, 20, 6)

    if st.button("Generate"):
        img = load_and_resize_image(tmp_path, size=(100, 100))
        pixels = img.reshape(-1, 3)

        if method == "Hard Palette":
            k = num_colors if not auto_colors else None
            km = fit_kmeans(pixels, max_k=(k or 20), use_elbow=auto_colors)
            palette = get_hex_codes_from_centers(km.cluster_centers_)
            st.subheader("Hard Palette")
            plot_swatch(palette)
            st.code(palette)
        elif method == "Soft Palette":
            k = num_colors if not auto_colors else None
            gmm = fit_gmm(pixels, max_k=(k or 20), use_bic=auto_colors)
            palette = get_hex_codes_from_gmm_means(gmm.means_)
            st.subheader("Hard Palette")
            plot_swatch(palette)
            st.code(palette)
        else:  
            k = num_colors if not auto_colors else 6
            st.subheader("Soft Mood Gradient")
            palette = gmm_soft_gradient(tmp_path, n_components=k)
            plot_wheel(palette)
            st.code(palette)

else:  
    st.subheader("Extracted Palette")
    palette = extract_palette_by_frequency_and_lab(tmp_path)
    plot_swatch(palette)
    st.code(palette)

    st.subheader("Transform Palette")
    task = st.selectbox("", ["Generate Gradient", "Generate Opposites"])

    if task == "Generate Gradient":
        steps = st.slider("Steps per transition", 10, 100, 50)
        grad = generate_full_hsl_gradient(palette, steps_per_transition=steps)
        st.subheader("Gradient Wheel")
        plot_wheel(grad)
    else:
        opposites = generate_opposite_palette(palette)
        st.subheader("Opposite Wheel")
        plot_wheel(opposites)


# ———————————————————————————————
#  CLEANUP
# ———————————————————————————————
os.remove(tmp_path)
