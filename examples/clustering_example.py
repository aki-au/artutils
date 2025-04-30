# from artutils.io_utils import load_and_resize_image
# from artutils.clustering import fit_kmeans, fit_gmm
# from artutils.color_utils import get_hex_codes_from_centers, get_hex_codes_from_gmm_means
# from artutils.visualization import plot_color_swatch

from io_utils import load_and_resize_image
from clustering import fit_kmeans, fit_gmm
from color_utils import get_hex_codes_from_centers, get_hex_codes_from_gmm_means
from visualization import plot_color_swatch
import matplotlib.pyplot as plt

image = load_and_resize_image("example_image.png", size=(100, 100))
pixels = image.reshape(-1, 3)

kmeans = fit_kmeans(pixels, max_k=8, use_elbow=False)
hex_codes_kmeans = get_hex_codes_from_centers(kmeans.cluster_centers_)
plot_color_swatch(hex_codes_kmeans, "kmeans_pal.png")


gmm = fit_gmm(pixels, max_k=10, use_bic=False)
hex_codes_gmm = get_hex_codes_from_gmm_means(gmm.means_)
plot_color_swatch(hex_codes_gmm, "gmm_pal.png")
