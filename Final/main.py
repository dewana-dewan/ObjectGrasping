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
#print(sys.path)

from rectangle_related import *
from histo import *
from image_proc import *
from modelling import *


def test () :
	model = readImageAndTrain ();

	for i in range(300, 397, 4):
		# try:
		print(i)
		i = str(i)
		path  = '../../03/03_25/pcd0' + i + 'r.png'
		img = cv2.imread (path)
		bw_img = cv2.imread(path, 0)


		X, complete_image = createRectangles(img, model)
		#print(len(X), len(X[0]), len(X[0][0]))

		print(model.classes_)
		q = PriorityQueue()

		for image in X :
			sobelEdge = sobelEdgeDetection(cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY))
			cannyEdge = cannyEdgeDetection (copy.deepcopy(image[0]))
			lawsMasks = applyLawsMask(copy.deepcopy(image[0]))

			lawsMasks.append(sobelEdge)
			lawsMasks.append(cannyEdge)

			fcs = focusImg(img)
			fcs_hist = build_histogram(fcs, 5)

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
			all_hist.extend(fcs_hist)

			for i in range(len(lawsMasks)):
				
				ding = np.matmul(lawsMasks[i], lawsMasks[i].transpose())
				# ding = lawsMasks[i]
				print(ding.max())
				if (ding.max() != 0):
					ding *= int(255.0/float(ding.max()))
				print(ding.max())


				rect_hist = build_histogram(ding, 5)
				all_hist.extend(rect_hist)

			print("Prediction Score")

			# plt.subplot(121),plt.imshow(complete_image)
			# plt.subplot(122),plt.imshow(image)
			# plt.show()

			# from sklearn.preprocessing import MinMaxScaler
			# from sklearn.preprocessing import StandardScaler
			# from sklearn.preprocessing import Normalizer
			# # scaler = MinMaxScaler(feature_range=(0, 1))
			# # X = scaler.fit_transform(X)
			# #print (all_hist[100])
			# scaler = MinMaxScaler(feature_range=(0, 1))
			# data = scaler.fit_transform(np.array([all_hist]));
			
			from sklearn.preprocessing import MinMaxScaler
			from sklearn.preprocessing import StandardScaler
			from sklearn.preprocessing import Normalizer
			# scaler = MinMaxScaler(feature_range=(0, 1))
			# X = scaler.fit_transform(X)
			tempArr = np.array([all_hist])
			scaler = StandardScaler().fit(tempArr)
			transformedArr = scaler.transform(tempArr)
			
			ans = model.predict_proba(np.array(all_hist))
			#	print(ans)
			print(ans.shape, len(image))
			q.put((ans[0][1], image))

	# readImageAndTrain()
		i = 6
		while (i >=0 and not q.empty()):
			i -= 1
			img = q.get()
			# print(img[0], i)
			# print(img)
			cv2.rectangle(complete_image,img[1][1][0], img[1][1][1], (0,(i*60)%255,(i*20)*100),1)
			pass
		#plt.imshow(complete_image)
		#plt.show()
		#plt.savefig('complete_image.png')
		cv2.imwrite('complete_image' + no +'.png', complete_image)

test()
# for i in range(300, 397, 4):
# 	try:
# 		test(str(i))
# 	except:
# 		print('some error')

# plotData()
# test()
