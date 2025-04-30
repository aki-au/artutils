import cv2, os
import matplotlib.pyplot as plt
from PIL import Image

def load_and_resize_image(path, size=(100, 100)):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, size) if size else img_rgb

def plot_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def save_tile(tile, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(tile).save(out_path)