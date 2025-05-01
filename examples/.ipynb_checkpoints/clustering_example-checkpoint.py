from artutils.io_utils import load_and_resize_image
from artutils.clustering import fit_kmeans, fit_gmm
from artutils.color_utils import get_hex_codes_from_centers, get_hex_codes_from_gmm_means
from artutils.visualization import plot_swatch


image  = load_and_resize_image("Landscape.png", size=(100, 100))
pixels = image.reshape(-1, 3)

#KMeans
kmeans = fit_kmeans(pixels, max_k=8,  use_elbow=False)
hex_codes_km = get_hex_codes_from_centers(kmeans.cluster_centers_)
plot_swatch(hex_codes_km, save_path="kmeans_pal.png")

#GMM
gmm = fit_gmm(pixels,    max_k=10, use_bic=False)
hex_codes_gmm = get_hex_codes_from_gmm_means(gmm.means_)
plot_swatch(hex_codes_gmm, save_path="gmm_pal.png")
