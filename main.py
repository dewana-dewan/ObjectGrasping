import cv2
import copy
import time
import scipy
from math import *
import numpy as np
from matplotlib import pyplot as plt


def plot_rectangles(file_name, img):

	pts = np.array([], np.int32)
	cnt = 1
	rectangles = []

	with open(file_name, 'r') as fl:
		for line in fl:

			content = line.strip().split(' ')
			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)

			if cnt % 4 == 0:				
				print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))
				
				print(atan(m), angl, c_x, c_y, ht, wt)
				cv2.circle(img, (c_x, c_y) , 1, (50,0,0))

				rect = subimage(img, (c_x, c_y), atan(m)*180/(np.pi), wt, ht)
				rectangles.append(rect)
				
				plt.subplot(111),plt.imshow(rect)
				plt.title('Rectangles'), plt.xticks([]), plt.yticks([])
				plt.show()
				
				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				print(pts[0][0])
				pts = np.array([], np.int32)
			cnt += 1

	return img


def subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)


def build_histogram(img_arr, bin_size):
    bins_arr = []
    bc = []

    for i in range(int(255/bin_size) + 1):
    	print(i)
    	bins_arr.append(0)
    	bc.append(i * bin_size)

    print(bins_arr)

    for i in range(img_arr.shape[0]):
    	for j in range(img_arr.shape[1]):
    		
    		a = int(img_arr[i][j] / bin_size)
    		b = img_arr[i][j] % bin_size
    		k = abs(0.5 - b / bin_size)

    		if (b <= bin_size / 2):
    			if (a == 0):
    				bins_arr[0] += 1
    			else:
    				bins_arr[a] += 1 - k
    				bins_arr[a - 1] += k
    		else:
    			bins_arr[a] += 1 - k
    			bins_arr[a + 1] += k
    			
    c = 0
    for i in range(int(255/bin_size) + 1):
    	c += bins_arr[i]

    print(bins_arr, c)
    return bins_arr, bc


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


def sobelEdgeDetection(img):

	kernel = np.ones((5,5),np.float32)/25
	img = cv2.filter2D(img,-1,kernel)

	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

	return sobely + sobelx