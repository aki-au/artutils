import colorsys, numpy as np
from skimage import color as skcolor

def hex_to_rgb_norm(h):
    h = h.lstrip('#'); return tuple(int(h[i:i+2],16)/255 for i in (0,2,4))

def rgb_norm_to_hex(rgb):
    r,g,b = [int(round(x*255)) for x in rgb]; return f"#{r:02x}{g:02x}{b:02x}"

def hex_to_lab(h):
    return skcolor.rgb2lab(np.array(hex_to_rgb_norm(h)).reshape(1,1,3))[0,0]

def delta_e(l1,l2):
    return np.linalg.norm(l1-l2)

def interpolate_hsl(a,b,steps=10):
    h1,l1,s1 = colorsys.rgb_to_hls(*hex_to_rgb_norm(a))
    h2,l2,s2 = colorsys.rgb_to_hls(*hex_to_rgb_norm(b))
    if abs(h2-h1)>0.5: h1-=1 if h1>h2 else 0; h2+=1 if h2<h1 else 0
    grad=[]
    for i in range(steps):
        h=(h1+i*(h2-h1)/(steps-1))%1.0
        l=l1+i*(l2-l1)/(steps-1)
        s=s1+i*(s2-s1)/(steps-1)
        grad.append(rgb_norm_to_hex(colorsys.hls_to_rgb(h,l,s)))
    return grad

def opposite_palette(pal):
    res=[]
    for h in pal:
        r,g,b=hex_to_rgb_norm(h); H,L,S=colorsys.rgb_to_hls(r,g,b)
        res.append(rgb_norm_to_hex(colorsys.hls_to_rgb((H+0.5)%1,L,S)))
    return res

def get_hex_codes_from_centers(centers):
    hex_codes = []
    for rgb in np.round(centers).astype(int):
        hex_code = '#{:02x}{:02x}{:02x}'.format(*rgb)
        hex_codes.append(hex_code)
    return hex_codes

def get_hex_codes_from_gmm_means(gmm_means):
    hex_codes = []
    for rgb in np.round(gmm_means).astype(int):
        hex_code = '#{:02x}{:02x}{:02x}'.format(*rgb)
        hex_codes.append(hex_code)
    return hex_codes