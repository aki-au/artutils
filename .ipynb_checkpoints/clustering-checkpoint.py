import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
from .color_utils import rgb_norm_to_hex

# k‑means helpers

def fit_kmeans(arr,max_k=11,use_elbow=False):
    if use_elbow:
        inertias=[KMeans(k,random_state=42,n_init='auto').fit(arr).inertia_ for k in range(1,max_k+1)]
        k=KneeLocator(range(1,max_k+1),inertias,curve='convex',direction='decreasing').knee or max_k
    else: k=max_k
    model=KMeans(k,random_state=42,n_init='auto').fit(arr); return model

def kmeans_centers_to_hex(centers):
    return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r,g,b in np.round(centers)]

# GMM helpers + soft gradient

def fit_gmm(arr,max_k=20,use_bic=False):
    if use_bic:
        bics=[GaussianMixture(k,random_state=42).fit(arr).bic(arr) for k in range(1,max_k+1)]
        k= np.argmin(bics)+1
    else: k=max_k
    return GaussianMixture(k,random_state=42).fit(arr)

def gmm_means_to_hex(means):
    return [f"#{int(r):02x}{int(g):02x}{int(b):02x}" for r,g,b in np.round(means)]

def gmm_soft_gradient(image_path,n_components=5,sample=5000,steps=30,dedup=5):
    from .io_utils import load_and_resize_image
    from .palette_tools import deduplicate_colors, sort_palette_by_closeness
    from .color_utils import interpolate_hsl, rgb_norm_to_hex
    img=load_and_resize_image(image_path,size=(100,100)).reshape(-1,3)
    if len(img)>sample: img=img[np.random.choice(len(img),sample,False)]
    gmm=GaussianMixture(n_components,random_state=42).fit(img)
    probs=gmm.predict_proba(img)
    blended=[rgb_norm_to_hex(np.average(gmm.means_/255,axis=0,weights=p)) for p in probs]
    pal=deduplicate_colors(blended,dedup)
    pal=sort_palette_by_closeness(pal)
    grad=[]
    for i in range(len(pal)):
        grad+=interpolate_hsl(pal[i],pal[(i+1)%len(pal)],steps)[:-1]
    return grad
