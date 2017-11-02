import numpy as np
import cv2
from matplotlib import pyplot as plt

def div( a, b ):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide( a, b )
        c[ ~ np.isfinite( c )] = 0  # -inf inf NaN
    return c


def showImage (img) :
    cv2.imshow('image', img);
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('newImg.png',img)
        cv2.destroyAllWindows()

def applyLawsMask (Y) :
    L5 = [[1], [4], [6], [4], [1]]
    E5 = [[-1], [-2], [0], [2], [1]]
    S5 = [[-1], [0], [2], [0], [-1]]
    R5 = [[1], [-4], [6], [-4], [1]]
    imgs = []
    kernel = np.ones((15,15),np.float32)/225
    #L5E5
    L5E5 = cv2.filter2D(Y,-1,(L5@np.transpose(E5)))
    #L5E5 = L5E5 + cv2.filter2D(L5E5,-1,kernel)
    #E5L5
    E5L5 = cv2.filter2D(Y,-1,(E5@np.transpose(L5)))
    #E5L5 = E5L5 + cv2.filter2D(E5L5,-1,kernel)
    #L5R5
    L5R5 = cv2.filter2D(Y,-1,(L5@np.transpose(R5)))
    #L5R5 = L5R5 + cv2.filter2D(L5R5,-1,kernel)
    #R5L5
    R5L5 = cv2.filter2D(Y,-1,(R5@np.transpose(L5)))
    #R5L5 = R5L5 + cv2.filter2D(R5L5,-1,kernel)
    #E5S5
    E5S5 = cv2.filter2D(Y,-1,(E5@np.transpose(S5)))
    #E5S5 = E5S5 + cv2.filter2D(E5S5,-1,kernel)
    #S5E5
    S5E5 = cv2.filter2D(Y,-1,(S5@np.transpose(E5)))
    #S5E5 = S5E5 + cv2.filter2D(S5E5,-1,kernel)
    #S5S5
    S5S5 = cv2.filter2D(Y,-1,(S5@np.transpose(S5)))
    #S5S5 = S5S5 + cv2.filter2D(S5S5,-1,kernel)
    #R5R5
    R5R5 = cv2.filter2D(Y,-1,(R5@np.transpose(R5)))
    #R5R5 = R5R5 + cv2.filter2D(R5R5,-1,kernel)
    #L5S5
    L5S5 = cv2.filter2D(Y,-1,(L5@np.transpose(S5)))
    #L5S5 = L5S5 + cv2.filter2D(L5S5,-1,kernel)
    #S5L5
    S5L5 = cv2.filter2D(Y,-1,(S5@np.transpose(L5)))
    #S5L5 = S5L5 + cv2.filter2D(S5L5,-1,kernel)
    #E5E5
    E5E5 = cv2.filter2D(Y,-1,(E5@np.transpose(E5)))
    #L5E5 = L5E5 + cv2.filter2D(L5E5,-1,kernel)
    #E5R5
    E5R5 = cv2.filter2D(Y,-1,(E5@np.transpose(R5)))
    #E5R5 = E5R5 + cv2.filter2D(E5R5,-1,kernel)
    #R5E5
    R5E5 = cv2.filter2D(Y,-1,(R5@np.transpose(E5)))
    #R5E5 = R5E5 + cv2.filter2D(R5E5,-1,kernel)
    #S5R5
    S5R5 = cv2.filter2D(Y,-1,(S5@np.transpose(R5)))
    #S5R5 = S5R5 + cv2.filter2D(S5R5,-1,kernel)
    #R5S5
    R5S5 = cv2.filter2D(Y,-1,(R5@np.transpose(S5)))
    #R5S5 = R5S5 + cv2.filter2D(R5S5,-1,kernel)

    image = (E5L5 + L5E5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    image = (R5L5 + L5R5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    image = (E5S5 + S5E5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    image = S5S5/1.0
    image *= 255.0/image.max();
    imgs.append (image);

    image = R5R5/1.0
    image *= 255.0/image.max();
    imgs.append (image);

    image = (L5S5 + S5L5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    image = E5E5/1.0
    image *= 255.0/image.max();
    imgs.append (image)

    image = (R5E5 + E5R5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    image = (S5R5 + R5S5)/2
    image *= 255.0/image.max();
    imgs.append (image);

    return imgs


img = cv2.imread('./samples/pcd0312r.png')
#Convert to YCrCb
imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


rows, cols, channel = imgYCC.shape
#print (rows, cols, channel);

Y, Cr, Cb = cv2.split(imgYCC)
#Filter Image using average filter
kernel = np.ones((15,15),np.float32)/225
meanImg = cv2.filter2D(Y,-1,kernel)

#take difference meanImg - Y
YDiff = Y - meanImg;
lawsImgs = applyLawsMask (YDiff)

for i in range(9) :
    plt.subplot(3,3,i+1),plt.imshow(lawsImgs[i],cmap = 'gray')
plt.show()
