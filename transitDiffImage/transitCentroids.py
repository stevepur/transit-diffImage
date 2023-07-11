import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def quick_flux_centroid(arr, extent, constrain=True):
    xpix = np.linspace(extent[0], extent[1]-1, arr.shape[1])
    ypix = np.linspace(extent[2], extent[3]-1, arr.shape[0])
    X, Y = np.meshgrid(xpix, ypix)
    normArr = arr.copy() - np.median(arr.ravel())
    sum_f = np.sum(normArr.ravel())
    sum_x = np.sum((X*normArr).ravel())
    sum_y = np.sum((Y*normArr).ravel())
    
    xc = sum_x/sum_f
    yc = sum_y/sum_f
    
    if constrain:
        # if the centroid is outside the extent then return the center of the image
        if (xc < extent[0]) | (xc > extent[1]):
            xc = np.mean(extent[0:2])

        if (yc < extent[2]) | (yc > extent[3]):
            yc = np.mean(extent[2:])

    return [xc, yc]

def render_prf(prf, coef, cData):
    # coef = [x, y, a, o]
    return coef[3] + coef[2]*prf.evaluate(coef[0] - cData["dCol"] + 0.5, coef[1] - cData["dRow"] + 0.5)
    
def sim_image_data_diff(c, prf, cData, data):
    pix = render_prf(prf, c, cData)
    return np.sum((pix.ravel() - data.ravel())**2)
    
def tess_PRF_centroid(prf, prfExtent, diffImage, catalogData):
    data = diffImage.copy()

    xOffset = int(np.round(catalogData["ticColPix"][0]%1))
    yOffset = int(np.round(catalogData["ticRowPix"][0]%1))

    closeData = data[7+yOffset:14+yOffset, 7+xOffset:14+xOffset]

    closeExtent = [prfExtent[0]+7+xOffset, prfExtent[1]-7+xOffset, prfExtent[2]+7+yOffset, prfExtent[3]-7+yOffset]

    qfc = quick_flux_centroid(closeData, closeExtent)

    seed = np.array([qfc[0], qfc[1], 1, 0])
    data = diffImage.copy()
    r = minimize(sim_image_data_diff, seed, method="L-BFGS-B",
                 args = (prf, catalogData, data))
                 
    simData = render_prf(prf, r.x, catalogData)

    prfFitQuality = np.corrcoef(simData.ravel(), data.ravel())[0,1]
    
    return r.x, prfFitQuality, qfc, closeData, closeExtent
