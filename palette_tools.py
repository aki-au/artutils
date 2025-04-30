import numpy as np
from .color_utils import hex_to_lab, delta_e
from .io_utils import load_and_resize_image
from skimage import color as skcolor

def deduplicate_colors(colors,thr=5):
    labs=[hex_to_lab(c) for c in colors]; keep=[]
    for i,c in enumerate(colors):
        if all(delta_e(labs[i],labs[j])>thr for j in keep): keep.append(i)
    return [colors[i] for i in keep]

def sort_palette_by_closeness(pal):
    if not pal: return []
    labs=[hex_to_lab(c) for c in pal]
    used=[False]*len(pal); order=[0]; used[0]=True
    for _ in range(len(pal)-1):
        last=order[-1]
        nxt=min((i for i,u in enumerate(used) if not u), key=lambda j:delta_e(labs[last],labs[j]))
        order.append(nxt); used[nxt]=True
    return [pal[i] for i in order]

def extract_palette_frequency_lab(path,resize=(300,300),min_px=150,dE=5,max_colors=None):
    img=load_and_resize_image(path,size=resize).reshape(-1,3)
    cols,counts=np.unique(img,axis=0,return_counts=True)
    order=np.argsort(-counts)
    final=[]; labs=[]
    for rgb in cols[order]:
        if counts[order[0]]<min_px: break
        lab=skcolor.rgb2lab((rgb/255).reshape(1,1,3))[0,0]
        if not labs or all(np.linalg.norm(lab-l)>dE for l in labs):
            final.append(rgb); labs.append(lab)
            if max_colors and len(final)>=max_colors: break
    return [f"#{r:02x}{g:02x}{b:02x}" for r,g,b in final]
