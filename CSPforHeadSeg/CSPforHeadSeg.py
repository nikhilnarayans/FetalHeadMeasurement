from skimage import io
import numpy as np
import cv2 as cv
from cv2 import linearPolar, WARP_FILL_OUTLIERS, WARP_INVERSE_MAP

pInputImage = io.imread("F:\ISBI_Challenge\FetalHeadSeg\head3.jpg",as_grey=True)

nX = np.size(pInputImage,0)
nY = np.size(pInputImage,1)

nMaxRadius = np.sqrt(nX**2+nY**2)/2

pTransformImg = linearPolar(pInputImage,(nX/2,nY/2),nMaxRadius, flags=WARP_FILL_OUTLIERS)

io.imshow(pTransformImg.T)
io.show()