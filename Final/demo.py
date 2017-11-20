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
import random


def displayRectangle (file_name, img):
	pts = np.array([], np.int32)
	cnt = 0
	rectangles = []
	k = 1;
	flag = False
	skip = 0;
	
	xDir = random.randint (-30, 30)
	yDir = random.randint (-30, 30)
	
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
			


			pt = [int(float(content[0])) + xDir, int(float(content[1])) + yDir]
			pts = np.append(pts, pt)
			#print(pts)

			if cnt % 4 == 0:
				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				plt.imshow(img)
				plt.show()
				#print(pts[0][0])
				pts = np.array([], np.int32)
				xDir = random.randint (-10, 10)
				yDir = random.randint (-10, 10)

	return

def readImage (image) :
	plt.subplot(111), plt.imshow(image);
	plt.show();

def main() :
	imageName = sys.argv[1];
	image = cv2.imread(imageName)
	plt.imshow(image)
	plt.show()

	goodRectangleFileName = ""
	goodRectangleFileName = imageName.split('.png')[0][:-1] + 'cpos.txt'
	displayRectangle (goodRectangleFileName, image)

main()