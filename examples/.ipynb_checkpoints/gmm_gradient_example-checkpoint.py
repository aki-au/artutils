from artist_toolbox.clustering import generate_soft_gmm_color_wheel

generate_soft_gmm_color_wheel(
    image_path="example_image.png",
    n_components=6,
    sample_size=5000,
    steps_per_transition=40,
    deduplication_threshold=5,
    figsize=(10, 10)
)
