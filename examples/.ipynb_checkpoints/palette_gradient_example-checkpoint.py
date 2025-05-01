from artutils.palette_tools import extract_palette_by_frequency_and_lab
from artutils.visualization import plot_swatch
from artutils.color_utils import generate_opposite_palette, generate_full_hsl_gradient

#Extract the palette from a picture
palette = extract_palette_by_frequency_and_lab("swatch.png")
plot_swatch(palette)

#Generate the opposite colors
opposites = generate_opposite_palette(palette)
plot_swatch(opposites)

#Generate the whole gradient from a palette
full_grad = generate_full_hsl_gradient(palette, steps_per_transition=100)
plot_wheel(full_grad)
