from artutils.visualization import plot_color_swatch, plot_color_wheel_gradient
from artutils.color_utils import generate_full_hsl_gradient

palette = ['#123456', '#abcdef', '#fffffa', '#654321']
plot_swatch(palette)
gradient = generate_full_hsl_gradient(palette, steps_per_transition=100) 
plot_wheel(gradient)