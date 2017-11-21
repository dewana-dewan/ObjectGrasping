import _pickle as cPickle
import numpy as np
import pandas as pd
import cv2
import copy


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


def applyLawsMask (img) :
	#return []
	imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)


	rows, cols, channel = imgYCC.shape
	#print (rows, cols, channel);

	Y, Cr, Cb = cv2.split(imgYCC)
	#Filter Image using average filter
	kernel = np.ones((15,15),np.float32)/225
	meanImg = cv2.filter2D(Y,-1,kernel)

	#take difference meanImg - Y
	YDiff = Y - meanImg;
	Y = YDiff

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
	image *= 225.0/image.max();
	imgs.append (image);

	image = (R5L5 + L5R5)/2
	image *= 225.0/image.max();
	imgs.append (image);

	image = (E5S5 + S5E5)/2
	image *= 225.0/image.max();
	imgs.append (image);

	image = S5S5/1.0
	image *= 225.0/image.max();
	imgs.append (image);

	image = R5R5/1.0
	image *= 225.0/image.max();
	imgs.append (image);

	image = (L5S5 + S5L5)/2
	image *= 225.0/image.max();
	imgs.append (image);

	image = E5E5/1.0
	image *= 225.0/image.max();
	imgs.append (image)

	image = (R5E5 + E5R5)/2
	image *= 225.0/image.max();
	imgs.append (image);

	image = (S5R5 + R5S5)/2
	image = image/1.0
	image *= 225.0/image.max();
	imgs.append (image);

	return imgs


def sobelEdgeDetection(img):

	# kernel = np.ones((5,5),np.float32)/25
	# img = cv2.filter2D(img,-1,kernel)

	# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	# image = sobelx + sobely
	# image = image/1.0
	# image *= 25.0/image.max()
	# plt.subplot(111),plt.imshow(sobely)
	# plt.show()
	image = cv2.Sobel (img, cv2.CV_8U, 1, 0, ksize = 5)
	return image

def cannyEdgeDetection(img):
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bw_edges = cv2.Canny(gray,50,100)

	# kernel = np.ones((3, 3), dtype=np.float64)
	# dilated = cv2.morphologyEx(bw_edges, cv2.MORPH_DILATE, kernel)
	# dilated = dilated/1.0
	# dilated *= 1.0/dilated.max()

	# plt.subplot(111),plt.imshow(dilated)
	# plt.show()

	return bw_edges
