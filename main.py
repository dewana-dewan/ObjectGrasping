import cv2
import copy
import time
import scipy
from math import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC

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
				# print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))

				# print(atan(m), angl, c_x, c_y, ht, wt)
				cv2.circle(img, (c_x, c_y) , 1, (50,0,0))

				rect = subimage(img, (c_x, c_y), atan(m)*180/(np.pi), wt, ht)
				rectangles.append(rect)

				plt.subplot(111),plt.imshow(rect)
				plt.title('Rectangles'), plt.xticks([]), plt.yticks([])
				plt.show()

				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				# print(pts[0][0])
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
		#print(i)
		bins_arr.append(0)
		# bc.append(i * bin_size)

	# print(bins_arr)

	for i in range(img_arr.shape[0]):
		for j in range(img_arr.shape[1]):

			a = int(img_arr[i][j] / bin_size)
			b = img_arr[i][j] % bin_size
			k = abs(0.5 - b / bin_size)
			# print(a, b, img_arr[i][j])

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
	for i in range(int(255/bin_size) + 1):
		bins_arr[i] /= c

	#print(bins_arr, c)
	return bins_arr


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
	image = sobelx + sobely
	image = image/image.max()
	# plt.subplot(111),plt.imshow(sobely)
	# plt.show()

	return image

def cannyEdgeDetection(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	bw_edges = cv2.Canny(gray,50,100)

	kernel = np.ones((3, 3), dtype=np.float64)
	dilated = cv2.morphologyEx(bw_edges, cv2.MORPH_DILATE, kernel)


	dilated = dilated/dilated.max()

	# plt.subplot(111),plt.imshow(dilated)
	# plt.show()

	return dilated

# background elimination

def getRectangles(file_name, lawsMasks):

	pts = np.array([], np.int32)
	cnt = 1
	rectangles = []

	with open(file_name, 'r') as fl:
		for line in fl:

			this_rectangle = []
			content = line.strip().split(' ')
			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)
			#print(pts)

			if cnt % 4 == 0:
				# print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))

				#print(atan(m), angl, c_x, c_y, ht, wt)
				pts = np.array([], np.int32)

				k = 0

				for img in lawsMasks:
					rect = subimage(img, (c_x, c_y), atan(m)*180/(np.pi), wt, ht)
					# plt.subplot(3,4,k+1),plt.imshow(rect)
					# k += 1
					# # plt.subplot(111),plt.imshow(rect)
					# plt.title(str(k)), plt.xticks([]), plt.yticks([])
					this_rectangle.append(rect)

				plt.show()
				rectangles.append(this_rectangle)

				# cv2.circle(img, (c_x, c_y) , 1, (50,0,0))
				# pts = pts.reshape((-1,1,2))
				# cv2.polylines(img, [pts], True, (100,0,0), 1)
				# print(pts[0][0])
				# pts = np.array([], np.int32)

			cnt += 1

	return rectangles


def trainingSVM (X, y):
	X = np.array(X)
	y = np.array(y)
	classifier_conf = SVC(kernel='linear',C = 1, gamma = 'auto', probability=True)
	classifier_conf.fit(X, y)

	for i in range(10) :
		x = [X[i]]
		print (classifier_conf.predict_proba(np.array(x)));

	return



def main():
	img_no = 312
	img = cv2.imread('./samples/pcd0' + str(img_no) + 'r.png')
	bw_img = cv2.imread('./samples/pcd0' + str(img_no) + 'r.png', 0)

	sobelEdge = sobelEdgeDetection(copy.deepcopy(bw_img))
	cannyEdge = cannyEdgeDetection(copy.deepcopy(img))
	lawsMasks = applyLawsMask(copy.deepcopy(img))


	lawsMasks.append(sobelEdge)
	lawsMasks.append(cannyEdge)

	# for i in range(11) :
	# 	plt.subplot(3,4,i+1),plt.imshow(lawsMasks[i],cmap = 'gray')
	# plt.show()

	goodRectangles = getRectangles('./samples/pcd0' + str(img_no) + 'cpos.txt', lawsMasks)
	#print(len(goodRectangles[0]))
	#return;
	X = []
	for aRectangle in goodRectangles:
		bc = []
		bin_size = 30
		all_hists = []
		# for j in range (11) :
		# 	for i in range(len(aRectangle) * int(255/bin_size) + 1):
		# 		# bins_arr.append(0)
		# 		print(i)
		# 		bc.append(i * bin_size)

		for features in aRectangle:
			# print(features)
			rect_hist = build_histogram(features, 30)
			all_hists.extend(rect_hist)
		# x = np.linspace(1, 99, 99)
		# plt.plot(x, all_hists, lw=2)
		# plt.show()
		#print(len(all_hists), all_hists)
	# print(lawsMasks[10], len(lawsMasks))
		X.append(all_hists);

	#print (len(X))
	y = []
	for i in range(5) :
		y.append(1)
	for i in range(5) :
		y.append(0)

	trainingSVM (X, y)
	# print(goodRectangles)

main()