#%**************************************************************************
#% AUTOMATIC FETAL HEAD MEASUREMENTS FROM ULTRASOUND IMAGES USING CIRCULAR
#% SHORTEST PATHS
#%
#% Developer:    Nikhil Narayan S
#%
#% Date:         05/02/2016
#% 
#% Description: This file contains the implementation of the algorithm 
#%              proposed in [1] as a part of the ISBI-2012 challenge. 
#%              It was shown in [2] that this algorithm outperformed
#%              other methods that were submitted for the challenge to
#%              measure the biparietal diameter (BPD) and the occipito-
#%              frontal diameter (OFD).
#%
#%              [1]. Sun, Changming. "Automatic fetal head measurements 
#%                   from ultrasound images using circular shortest 
#%                   paths." Proceedings of Challenge US: Biometric 
#%                   Measurements from Fetal Ultrasound Images, ISBI 2012 
#%                   (2012): 13-15.
#%
#%              [2]. Rueda, Sylvia, et al. "Evaluation and comparison of 
#%                   current fetal ultrasound image segmentation methods 
#%                   for biometric measurements: a grand challenge." 
#%                   IEEE Transactions on Medical Imaging, 33.4 (2014): 
#%                   797-813.
#%**************************************************************************

import Tkinter as tk
from tkFileDialog import askopenfilename
from skimage import io, graph
from skimage.measure import EllipseModel
import numpy as np
from cv2 import linearPolar, WARP_FILL_OUTLIERS, WARP_INVERSE_MAP
from numpy import Inf
from EllipsePackage import *
from CSP import *

root = tk.Tk()
root.withdraw()
filename = askopenfilename()
print(filename)

pInputImage = io.imread(filename,as_grey=True)

nX = np.size(pInputImage,0)
nY = np.size(pInputImage,1)

nMaxRadius = np.sqrt(nX**2+nY**2)/2

#-----------------------------------------------------------------
# Transform image into polar co-ordinates
#-----------------------------------------------------------------

pTransformImg = linearPolar(pInputImage,(nX/2,nY/2),nMaxRadius, flags=WARP_FILL_OUTLIERS)
pTransformImg = pTransformImg.T

#-----------------------------------------------------------------
# Apply CSP to segment fetal skull
#-----------------------------------------------------------------

pTransformImg = CSP(pTransformImg)

#-----------------------------------------------------------------
# Transform back to cartesian co-ordinates
#-----------------------------------------------------------------

pTransformImg = linearPolar(pTransformImg.T,(nX/2,nY/2),nMaxRadius, flags=WARP_INVERSE_MAP)

#-----------------------------------------------------------------
# Perform head measurements
#-----------------------------------------------------------------

pPoints = np.argwhere(pTransformImg==1.0)

#-----------------------------------------------------------------
# Off the shelf algorithm employed to fit an ellipse
#-----------------------------------------------------------------
a = fitEllipse(pPoints[:,0],pPoints[:,1])
center = ellipse_center(a)
phi = ellipse_angle_of_rotation2(a)
axes = ellipse_axis_length(a)

print "OFD = ", axes[0], "pixels"
print "BPD = ", axes[1], "pixels"

#-----------------------------------------------------------------
# Display segmented skull
#-----------------------------------------------------------------
io.imshow(pTransformImg)
io.show()