import matplotlib.pyplot as plt, matplotlib.patches as patches, numpy as np
from .palette_tools import sort_palette_by_closeness

def plot_swatch(pal):
    fig,ax=plt.subplots(figsize=(len(pal),2))
    for i,c in enumerate(pal): ax.add_patch(patches.Rectangle((i,0),1,1,color=c))
    ax.set_xlim(0,len(pal)); ax.set_ylim(0,1); ax.axis('off'); plt.tight_layout(); plt.show()

def plot_wheel(pal,inner=0.5,width=0.5,figsize=(8,8)):
    pal=sort_palette_by_closeness(pal); n=len(pal); ang=np.linspace(0,2*np.pi,n,endpoint=False)
    fig,ax=plt.subplots(figsize=figsize,subplot_kw={'polar':True}); ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    for a,c in zip(ang,pal): ax.bar(a,width,2*np.pi/n,bottom=inner,color=c,linewidth=0)
    ax.axis('off'); plt.tight_layout(); plt.show()