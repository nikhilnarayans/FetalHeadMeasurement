import numpy as np
from skimage import io

def CSP(pInputImage):
    nX = np.size(pInputImage,0)
    nY = np.size(pInputImage,1)
    nMaxVal = np.max(pInputImage,(0,1))
    pInputLastCol = pInputImage[:,nY-1].tolist()
    
    k = 145 #int(nX/3)

    pPathImage = np.copy(pInputImage)
    pPathImage = nMaxVal-pPathImage
    d = np.ones([nX,nY+1])*np.Inf
    d[k,0] = 0;
    p = np.empty([nX,nY+1])
    pPathImage[:,nY-1] = np.Inf
    pPathImage[k-1:k+2,nY-1] = pInputLastCol[k-1:k+2]
    for j in xrange(1,nY+1,1):
        if j!=nY+1:
            for i in xrange(1,nX-1,1):
                pPoints = [i-1,i,i+1]
                pVals = [d[i-1,j-1]+pPathImage[i-1,j-1],d[i,j-1]+pPathImage[i,j-1],d[i+1,j-1]+pPathImage[i+1,j-1]]
                d[i,j] = np.min(pVals)
                p[i,j] = pPoints[np.argmin(pVals)]
                                
    nRow = np.argmin(d[:,nY])
    nRowPrev = p[nRow,nY]
    for j in xrange(nY-1,0,-1):
        pInputImage[nRowPrev,j] = 1.0
        nRowPrev = p[nRowPrev,j]

    #io.imshow(pInputImage)
    #io.show()
    return pInputImage

        #print d[:,nY]

