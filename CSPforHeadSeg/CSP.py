#%**************************************************************************
#% SEGMENTING THE FETAL SKULL USING CIRCULAR SHORTEST PATH ALGORITHM
#%
#% Developer:    Nikhil Narayan S
#%
#% Date:         05/02/2016
#% 
#% Description: This method segments the skull in theinput image using the 
#%              circular shortest path (CSP) algorithm proposed in [1]. It 
#%              is assumed that the input image is transformed into polar
#%              co-ordinates. The multiple search algorithm is used in 
#%              this implementation.
#%
#%              [1]. Sun, Changming, and Stefano Pallottino. "Circular 
#%                   shortest path in images." Pattern Recognition 36.3 
#%                   (2003): 709-719.
#%
#% Usage: outputImage = CSP(inputImage);
#% 
#% Where: inputImage  = Input polar transformed image in the range [0,1]
#%        outputImage = Segmented image in polar co-ordinates
#%
#%**************************************************************************

import numpy as np
from skimage import io

def CSP(pInputImage):
    nX = np.size(pInputImage,0)
    nY = np.size(pInputImage,1)
    nMaxVal = np.max(pInputImage,(0,1))
    pInputLastCol = pInputImage[:,nY-1].tolist()

    #-----------------------------------------------------------------
    # Input the approximate row at which the fetal head is present
    # For a complete implementation of the multiple search algorithm,
    # include the lines of code to follow in a loop iterated over the
    # rows of the image. The path that has the lowest cost for all
    # the rows gives the shortest path in the image. 
    # ## for k in xrange(1,nX,1):
    #-----------------------------------------------------------------

    k = 145 

    pPathImage = np.copy(pInputImage)

    #-----------------------------------------------------------------
    # Invert the image intensities to get better results. The skull is 
    # hyper-echoic and hence will cause the cost to increase if not 
    # inverted.
    #-----------------------------------------------------------------

    pPathImage = nMaxVal-pPathImage     
    d = np.ones([nX,nY+1])*np.Inf
    d[k,0] = 0;
    p = np.empty([nX,nY+1])

    #-----------------------------------------------------------------
    # Set large values to the pixels of the last column except for the  
    # neighbours in the 1st column. This is a crucial step of the CSP 
    # algorithm. Refer to [1] for more details.
    #-----------------------------------------------------------------

    pPathImage[:,nY-1] = np.Inf
    pPathImage[k-1:k+2,nY-1] = pInputLastCol[k-1:k+2]

    #-----------------------------------------------------------------
    # Apply Bellman's equation to determine the shortest path.
    # (Dynamic programming to determine shortest path in a graph)
    #-----------------------------------------------------------------

    for j in xrange(1,nY+1,1):
        if j!=nY+1:
            for i in xrange(1,nX-1,1):
                pPoints = [i-1,i,i+1]
                pVals = [d[i-1,j-1]+pPathImage[i-1,j-1],d[i,j-1]+pPathImage[i,j-1],d[i+1,j-1]+pPathImage[i+1,j-1]]
                d[i,j] = np.min(pVals)
                p[i,j] = pPoints[np.argmin(pVals)]
    
    #-----------------------------------------------------------------
    # Determine the shortest path and segment the pixels corresponding 
    # to the shortest path
    #-----------------------------------------------------------------
                           
    nRow = np.argmin(d[:,nY])
    nRowPrev = p[nRow,nY]
    pPoints = np.empty([nY,2])
    for j in xrange(nY-1,0,-1):
        pInputImage[nRowPrev,j] = 1.0
        nRowPrev = p[nRowPrev,j]

        #Co-ordinates of the shortest path. For future use.
        pPoints[j,:] = [nRowPrev,j]
    
    #-----------------------------------------------------------------
    # Return segmented image
    #-----------------------------------------------------------------

    return pInputImage



