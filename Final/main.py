import cv2
import copy
import time
import scipy
import sys
import os
import _pickle as cPickle
import numpy as np
import pandas as pd
from queue import PriorityQueue
from math import *
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates

sys.path.append(os.path.abspath('./'))
print(sys.path)

from rectangle_related import *
from histo import *
from image_proc import *
from modelling import *


def main (folderName, img_no):
	img = cv2.imread('../../'+ folderName + '/pcd' + img_no + 'r.png')
	print(os.path.abspath('../../'+ folderName + '/pcd' + img_no + 'r.png'))
	bw_img = cv2.imread('../../'+ folderName + '/pcd' + img_no + 'r.png', 0)

	sobelEdge = sobelEdgeDetection(copy.deepcopy(bw_img))
	cannyEdge = cannyEdgeDetection(copy.deepcopy(img))
	lawsMasks = applyLawsMask(copy.deepcopy(img))


	lawsMasks.append(sobelEdge)
	lawsMasks.append(cannyEdge)
	# for i in range(11) :
	# 	plt.subplot(3,4,i+1),plt.imshow(lawsMasks[i],cmap = 'gray')
	# plt.show()

	goodRectangles = getRectangles('../../'+ folderName + '/pcd' + img_no + 'cpos.txt', lawsMasks)
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
	badRectangles = getRectangles('../../'+ folderName + '/pcd' + img_no + 'cneg.txt', lawsMasks)
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

def test () :
	path  = '../../03/03_25/pcd0312r.png'
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

main('03', '0311')
# plotData()
# test()