from skimage import io, graph
from skimage.measure import EllipseModel
import numpy as np
from cv2 import linearPolar, WARP_FILL_OUTLIERS, WARP_INVERSE_MAP
from numpy import Inf
from EllipsePackage import *
from CSP import *

pInputImage = io.imread("F:\ISBI_Challenge\FetalHeadSeg\head4.jpg",as_grey=True)

nX = np.size(pInputImage,0)
nY = np.size(pInputImage,1)

nMaxRadius = np.sqrt(nX**2+nY**2)/2

pTransformImg = linearPolar(pInputImage,(nX/2,nY/2),nMaxRadius, flags=WARP_FILL_OUTLIERS)

pTransformImg = pTransformImg.T

pTransformImg = CSP(pTransformImg)

pTransformImg = linearPolar(pTransformImg.T,(nX/2,nY/2),nMaxRadius, flags=WARP_INVERSE_MAP)

io.imshow(pTransformImg)
io.show()

pPoints = np.argwhere(pTransformImg==1.0)

a = fitEllipse(pPoints[:,0],pPoints[:,1])
center = ellipse_center(a)
phi = ellipse_angle_of_rotation2(a)
axes = ellipse_axis_length(a)

print("center = ",  center)
print("angle of rotation = ",  phi)
print("axes = ", axes)