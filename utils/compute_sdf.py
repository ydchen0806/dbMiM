import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

def compute_sdf(img_gt):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """
    normalized_sdf = np.zeros_like(img_gt, dtype=np.float32)
    ids, counts = np.unique(img_gt, return_counts=True)
    # remove id 0
    if ids[0] == 0:
        ids = ids[1:]
    # if ids is None
    if len(ids) == 0:
        return normalized_sdf
    
    for id in ids:
        posmask = np.zeros_like(img_gt)
        posmask[img_gt == id] = 1
        posmask = posmask.astype(np.bool)
        if posmask.any():
            posdis = distance(posmask)
            posdis = (posdis - posdis.min()) / (posdis.max() - posdis.min())
            normalized_sdf += posdis
    return normalized_sdf
