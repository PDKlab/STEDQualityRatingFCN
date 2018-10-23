"""
Dataset creation from STED images.
THIS CODE IS PROVIDED AS A SAMPLE ONLY.
It is most likely that you will have to
adapt it depending on the format and range
of your input data, and the way the quality
scores are read.
"""
import os
import numpy as np
from skimage import io, filters
from collections import defaultdict

WITH_SCORES = True

def get_foreground(img):
    """Gets the foreground using a gaussian blur of sigma = 10 and the otsu threshold.

    :param img: A 2D numpy array

    :returns : A binary 2D numpy array of the foreground
    """
    blurred = filters.gaussian(img, sigma=10)
    blurred /= blurred.max()
    val = filters.threshold_otsu(blurred)
    return (blurred > val).astype('float32')

scores = defaultdict(float)
if WITH_SCORES:
    fscores = open("../quality", 'r')
    for line in fscores:
        if len(line) < 3:
            continue
        scores[int(line.split(",")[0].strip())] = float(line.split(",")[1].strip())


for f in os.listdir("."):                                                  
    if not "tif" in f:
        continue
    im = io.imread(f).astype(np.float32)
    im -= -float(2**15)
    im /= (float(2**15-1) - -float(2**15))

    mask=get_foreground(im)
    np.savez("dataset/"+str(int(f[:-5]))+"-{:.3f}.npz".format(scores[int(f[:-5])]), im, mask, mask)
