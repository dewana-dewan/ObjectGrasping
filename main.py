import cv2
import copy
import time
import scipy
import _pickle as cPickle
from queue import PriorityQueue
from math import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import cross_validation
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates


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
			# print (img_arr[i][j])
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
	cnt = 0
	rectangles = []
	k = 1;
	flag = False
	skip = 0;
	with open(file_name, 'r') as fl:
		for line in fl:
			cnt += 1
			this_rectangle = []
			content = line.strip().split(' ')

			if (flag and k == skip) :
				flag = False
				k = 1
				skip = 0
				continue
			if flag :
				k += 1
				continue

			if content[0] == 'NaN' or content[1] == 'NaN' :
				k = 1
				pts = np.array([], np.int32)
				if (cnt % 4 == 0) :
					continue
				else :
					skip = 4 - (cnt%4)
					flag = True
					continue

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
					rect = subimage(img, (c_x, c_y), angl, wt, ht)
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

	return rectangles

def saveModel (gnb) :
	with open('svmModel.pkl', 'wb') as fid:
	    cPickle.dump(gnb, fid)


def trainingSVM (X, y):

	X = np.array(X)
	y = np.array(y)
	#print(len(X), len(y))
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
	classifier_conf = SVC(kernel='linear',C = 1.0, gamma = 'auto', probability=True)
	classifier_conf.fit(X_train, y_train)

	print (classifier_conf.score(X_test, y_test))
	saveModel (classifier_conf)
	return classifier_conf


def plotData () :
	# X = np.array(X)
	# y = np.array(y)
	
	# np.savetxt("X10.csv", X, delimiter=",")
	# np.savetxt("y10.csv", y, delimiter=",")


	#cols = []
	#for i in range(0, )

	X = pd.io.parsers.read_csv('X10.csv');
	
	X = np.array(X)
	X = X.astype(np.float64, copy=False)
	
	y = pd.io.parsers.read_csv('y10.csv');
	
	y = np.array(y)
	y = y.astype(np.float64, copy=False)


	#print (X.isnull().any())
	#print (y.isnull().any())

	#X = np.array(X);
	#y = np.array(y);

	#print (X.shape)

	X_norm = (X - X.min())/(X.max() - X.min())

	print (X_norm)
	print (y.dtype)

	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	transformed = pd.DataFrame(pca.fit_transform(X_norm))

	plt.scatter(transformed[y==-1][0], transformed[y==-1][1], label='Negative Rectangles', c='red')
	plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Positive Rectangles', c='lightgreen')
	#plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

	plt.legend()
	plt.show()



def readImageAndTrain () :
	X = []
	Y = []
	# import os.path
	# if os.path.isfile('svmModel.pkl') :
	# 	with open('svmModel.pkl', 'rb') as fid:
	# 	    gnb_loaded = cPickle.load(fid)

	# 	return gnb_loaded

	for folderName in range(1, 11) :
		if (folderName == 9) :
			upto = 50
		elif folderName == 10 :
			upto = 35
		else :
			upto = 100
		name = str(folderName)
		if (folderName != 10) :
			name = '0' + name
		for i in range(2) :
			fname = name
			if (i == 1) :
				name = name + '/'+name+'_25'
			if (folderName == 2 and i == 1) :
				continue
			for img_no in range(1, upto) :
				if (i == 0 and img_no % 4 == 0) :
					continue
				if (i == 1 and img_no % 4 != 0) :
					continue;
				if (img_no < 10) :
					img_name = fname + '0' + str(img_no)
				else :
					img_name = fname + str(img_no)
				#print (name, img_name)
				print ('Doing folder', name, 'image number', img_name)
				x, y = main(name, img_name)
				X.extend(x)
				Y.extend(y)
				print ('Done folder', name, 'image number', img_name)
	#print (X)
	#print(Y)

	plotData (X, Y)

	#trainingSVM (X, Y)

def main (folderName, img_no):
	img = cv2.imread('../'+ folderName + '/pcd' + img_no + 'r.png')
	bw_img = cv2.imread('../'+ folderName + '/pcd' + img_no + 'r.png', 0)

	sobelEdge = sobelEdgeDetection(copy.deepcopy(bw_img))
	cannyEdge = cannyEdgeDetection(copy.deepcopy(img))
	lawsMasks = applyLawsMask(copy.deepcopy(img))


	lawsMasks.append(sobelEdge)
	lawsMasks.append(cannyEdge)
	# for i in range(11) :
	# 	plt.subplot(3,4,i+1),plt.imshow(lawsMasks[i],cmap = 'gray')
	# plt.show()

	goodRectangles = getRectangles('../'+ folderName + '/pcd' + img_no + 'cpos.txt', lawsMasks)
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

	Y = []
	badRectangles = getRectangles('../'+ folderName + '/pcd' + img_no + 'cneg.txt', lawsMasks)
	#print(len(badRectangles[0]))
	#return;
	for aRectangle in badRectangles:
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
		Y.append(all_hists);

	#print (len(X))
	y = []
	for i in range(len(X)) :
		y.append(1)
	for i in range(len(Y)) :
		y.append(-1)
	X.extend(Y)
	return X, y
	# print(goodRectangles)


def createRectangles(img):
	# print (img.shape)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print (gray.shape)
	bw_edges = cv2.Canny(gray,50,100)

	kernel = np.ones((3, 3), np.uint8)
	margin = 20
	dilated = cv2.morphologyEx(bw_edges, cv2.MORPH_DILATE, kernel)

	with_contours, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	max1 = 0
	max1_l = 0
	max2 = 0
	max2_l = 0

	for i in range(len(contours)):
		# print(i, len(contours[i]), max1, max2)
		# print()
		if max1_l <= len(contours[i]):
			max2_l = max1_l
			max1_l = len(contours[i])
			max2 = max1
			max1 = i
		elif max2_l <= len(contours[i]):
			max2_l = len(contours[i])
			max2 = i

	# print(max2, max1)

	x, y, width, height = cv2.boundingRect(contours[max2])
	roi = img[y - margin:y + height + margin, x - margin:x + width + margin]

	# roi containes the focused image now.
	# we'll now try and create rectangles

	# print(roi.shape[0], roi.shape[1])

	siz_x = int(roi.shape[0] / 5)
	siz_y = int(roi.shape[1] / 5)
	all_for_test = []

	for i in range(0, roi.shape[0] - siz_x, 10):
		for j in range(0, roi.shape[1] - siz_y, 10):

			local_roi = roi[j:j + siz_y, i:i + siz_x]
			bw_local_roi = cv2.cvtColor(local_roi,cv2.COLOR_BGR2GRAY)

			ret, thresh = cv2.threshold(bw_local_roi ,127,255,0)
			im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


			if ( len(contours) >= 1 and len(contours[0]) <= 4 ):
				print(len(contours[0]))
				continue;

			if ( len(contours) == 0):
				continue

			#cv2.rectangle(roi,(i, j), (i + siz_x, j + siz_y), (0,255,0),1)

			center = (i + siz_x/2, j + siz_y/2)
			# theta = -1 * 45
			theta = 0
			dings = []
			# ding = np.array([], np.int32)
			print('ding')
			# for k in range(3):
			ding = subimage(roi, center, theta, siz_x, siz_y)
			print (ding.shape, '------')
			print('ding')
			# dings.append(ding)
			# print (len(dings), '*********')
			theta += 45
			# return ;
			# plt.subplot(231),plt.imshow(roi)
			# plt.subplot(232),plt.imshow(local_roi)
			# plt.subplot(233),plt.imshow(thresh)
			#
			# plt.subplot(234),plt.imshow(dings[0])
			# # plt.subplot(235),plt.imshow(dings[1])
			# # plt.subplot(236),plt.imshow(dings[2])
			# plt.show()
			# return dings
			print(len(ding), 'akjdkjhaskdh')
			all_for_test.append(ding)
			print('all test length', len(all_for_test))
	return all_for_test


def test () :
	path  = '../03/03_25/pcd0312r.png'
	img = cv2.imread (path)
	bw_img = cv2.imread(path, 0)

	model = readImageAndTrain ();

	X = createRectangles(img)
	print(len(X), len(X[0]), len(X[0][0]))

	for image in X :
		sobelEdge = sobelEdgeDetection(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
		cannyEdge = cannyEdgeDetection (copy.deepcopy(image))
		lawsMasks = applyLawsMask(copy.deepcopy(image))

		lawsMasks.append(sobelEdge)
		lawsMasks.append(cannyEdge)

		# for image in lawsMasks :
		# 	plt.subplot(111),plt.imshow(image)
		# 	plt.show()

		# 	x = createRectangles (image)
		#
		# 	sobelEdge = sobelEdgeDetection(copy.deepcopy(bw_img))
		# 	cannyEdge = cannyEdgeDetection(copy.deepcopy(img))
		# 	lawsMasks = applyLawsMask(copy.deepcopy(img))
		#
		# 	lawsMasks.append(sobelEdge)
		# 	lawsMasks.append(cannyEdge)
		# 	print (len(x))
		# return ;
		# # q = PriorityQueue()
		# # print('testing now', X)
		all_hist= []
		for i in range(11):
			rect_hist = build_histogram(lawsMasks[0], 30)
			all_hist.extend(rect_hist)

		print("Prediction Score")
		ans = model.predict_proba(np.array([all_hist]))
		print(ans)

plotData()