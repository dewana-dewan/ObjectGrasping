import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates
from math import *
from matplotlib import pyplot as plt


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

def plotData () :
	# X = np.array(X)
	# y = np.array(y)
	
	# np.savetxt("X10.csv", X, delimiter=",")
	# np.savetxt("y10.csv", y, delimiter=",")


	#cols = []
	#for i in range(0, )

	X = pd.io.parsers.read_csv('../X10.csv');
	
	X = np.array(X)
	X = X.astype(np.float64, copy=False)
	
	y = pd.io.parsers.read_csv('../y10.csv');
	
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