import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

snscolors = np.array(sns.color_palette())
def pcacomponentplots(ax,vh,ylabels,colors=None):
    numev = vh.shape[0]
    xlim = 1.05*np.max(np.abs(vh))
    if colors is None:
        colors = np.tile([0.35,0.35,0.35],(numev,1))
    elif len(colors)<numev:
        colors = np.tile(colors,(numev,1))
    for evnum in range(numev):
        a=ax[evnum]
        x=vh[evnum]
        y = np.flipud(np.arange(vh.shape[1]))
    #     ax.plot(x,y,'-o',label='$\\vec e_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',c=snscolors[evnum])
        thickness=0.75
        a.barh(y+0*(evnum-1)*thickness,x,height=thickness,color=colors[evnum])
        a.set_yticks(y)
        a.set_yticklabels(ylabels,rotation='horizontal',fontsize=12)
        a.tick_params(labelsize=12)
        a.axvline(0,c='k',linestyle='-',linewidth=1)
#         a.set_title(label='$\\vec v_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',fontsize=14)
        a.set_xlim([-xlim,xlim])
        a.set_ylim([-0.5,len(y)-0.5])

def plot_tsne_withcolors(ax,tsne_result,quantity,title,corrskip=1,plotskip=1,colortype='scalar',qmin=0.001,qmax=0.999,alphaval=0.3,s=4,coloroffset=0,cmapname='cool',setxylimquantile=False):
    colordata = quantity.copy()
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
#         cmap=plt.get_cmap('Set1')
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=True)
    if setxylimquantile:
        ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
        ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    else:
        ax.set_xlim(np.quantile(tp[:,0],[0,1]))
        ax.set_ylim(np.quantile(tp[:,1],[0,1]))
    ax.set_title(title,fontsize=16)       
    return scatterplot, colordata    

def infer_angle_degrees(a_series: pd.Series) -> bool:
    """Heuristic: returns True if ANGLE looks like degrees, else radians."""
    a = a_series.to_numpy()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return False
    m = np.nanmax(np.abs(a))
    # If values exceed ~2π by a safe margin, assume degrees
    return m > 7.0

def safe_crop_with_padding(img, x0, y0, x1, y1, border_mode=cv2.BORDER_REPLICATE):
    """
    Crop a rectangle [y0:y1, x0:x1] from img, padding if it goes out of bounds.
    Returns an array of shape (y1-y0, x1-x0, 3).
    """
    H, W_img = img.shape[:2]
    # Desired size
    out_h = int(y1 - y0)
    out_w = int(x1 - x0)

    # Intersection with source
    sx0 = max(0, int(x0)); sy0 = max(0, int(y0))
    sx1 = min(W_img, int(x1)); sy1 = min(H, int(y1))

    # Extract intersecting part
    patch = img[sy0:sy1, sx0:sx1]

    # If exact size, return
    if (sx1 - sx0 == out_w) and (sy1 - sy0 == out_h):
        return patch

    # Need padding to reach desired size
    top    = sy0 - int(y0)
    left   = sx0 - int(x0)
    bottom = int(y1) - sy1
    right  = int(x1) - sx1

    patch_padded = cv2.copyMakeBorder(patch, top, bottom, left, right, border_mode)
    return patch_padded