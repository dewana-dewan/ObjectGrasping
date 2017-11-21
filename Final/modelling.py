import cv2
import os
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
from image_proc import *
from rectangle_related import *
from histo import *
# from ggplot import *
from sklearn.manifold import TSNE
import time


def draw_svm(X, y, C=1.0):
    plt.scatter(X[:,0], X[:,1], c=y)
    clf = SVC(kernel='poly', C=C, gamma = 'auto')
    
    clf_fit = clf.fit(X, y)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
                        alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], 
                clf.support_vectors_[:, 1], 
                s=100, linewidth=1, facecolors='none')
    plt.show()
    return clf

def plotData () :
	#print (X[0])
	#print (y)
	#X = np.array(X)
	#y = np.array(y)
	
	#np.savetxt("X.csv", X, delimiter=",")
	#np.savetxt("y.csv", y, delimiter=",")


	#cols = []
	#for i in range(0, )

	import csv
	X = []
	with open('../../processedData/x.txt') as csvfile:
	    readCSV = csv.reader(csvfile, delimiter=',')
	    for row in readCSV:
	        #print(row)
	        k = [float(s) for s in row[0].split(' ')]
	        X.append(k)
	        #print(row[0])
	        #print(row[0],row[1],row[2],)
	X = np.array(X)
	#X = X.astype(np.float64, copy=False)
	y = []
	with open('../../processedData/y.txt') as csvfile:
	    readCSV = csv.reader(csvfile, delimiter=',')
	    for row in readCSV:
	        #print(row)
	        k = int(row[0])
	        y.append(k)
	y = np.array(y)
	        #print(row[0])
	        #print(row[0],row[1],row[2],)
	#y = y.astype(np.float64, copy=False)


	#print (X.isnull().any())
	#print (y.isnull().any())

	#X = np.array(X);
	#y = np.array(y);

	#print (X.shape)

	#X_norm = (X - X.min())/(X.max() - X.min())

	#print (X_norm)
	#print (y.dtype)

	pca = sklearnPCA(n_components=2) #2-dimensional PCA
	transformed = pd.DataFrame(pca.fit_transform(X))

	print (transformed);

	plt.scatter(transformed[y==-1][0], transformed[y==-1][1], label='Negative Rectangles', c='red')
	plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Positive Rectangles', c='lightgreen')
	#plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Class 3', c='lightgreen')

	plt.legend()
	plt.show()

	transformed = np.array(transformed)

	#print (transformed)

	draw_svm (transformed, y)

# plotData()

def saveModel (gnb) :
	with open('svmModel.pkl', 'wb') as fid:
	    cPickle.dump(gnb, fid)




def cleanData (X, y) :
	X = np.array(X);
	y = np.array(y);
	
	print (X.shape, y.shape)

	feat_cols = [ 'x'+str(i) for i in range(X.shape[1]) ]
	df = pd.DataFrame(X,columns=feat_cols)
	df['label'] = y
	df['label'] = df['label'].apply(lambda i: str(i))

	X, y = None, None

	print (df.shape)
	rndperm = np.random.permutation(df.shape[0])

	n_sne = df.shape[0];

	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
	tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


	print ('t-SNE done! Time elapsed: seconds', format(time.time()-time_start))

	df_tsne = df.loc[rndperm[:n_sne],:].copy()
	df_tsne['x-tsne'] = tsne_results[:,0]
	df_tsne['y-tsne'] = tsne_results[:,1]

	chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) + geom_point(size=70,alpha=0.8) + ggtitle("tSNE dimensions colored by digit")
	print (chart)
	N = df.shape[0];
	b = np.column_stack((df_tsne['x-tsne'], df_tsne['y-tsne']))
	Y = np.array(df_tsne['label']);
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import Normalizer
	scaler = StandardScaler().fit(b)
	data = scaler.transform(b)
	#scaler = Normalizer().fit(b)
	#data = scaler.transform(b)
	return  draw_svm (data,Y)

	pca = sklearnPCA (n_components=3)
	pca_result = pca.fit_transform(df[feat_cols].values)

	df['pca-one'] = pca_result[:,1]
	df['pca-two'] = pca_result[:,2] 
	df['pca-three'] = pca_result[:,0]


	print ('Explained variation per principal component:', pca.explained_variance_ratio_)

	chart = ggplot( df.loc[rndperm[:700],:], aes(x='pca-one', y='pca-two', color='label') )+ geom_point(size=75,alpha=0.8)+ ggtitle("First and Second Principal Components colored by digit");
    
	print (chart)


def trainingSVM (X, y):
	from sklearn.preprocessing import MinMaxScaler
	from sklearn.preprocessing import StandardScaler
	from sklearn.preprocessing import Normalizer
	# scaler = MinMaxScaler(feature_range=(0, 1))
	# X = scaler.fit_transform(X)
	scaler = StandardScaler().fit(X)
	X = scaler.transform(X)
	#plotData (X, y)
	#return 
	# scaler = Normalizer().fit(X)
	# X = scaler.transform(X)
	# X = np.array(X)
	# y = np.array(y)
	# print (X.shape)
	# print (y.shape)
	# return 
	#print(len(X), len(y))
	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.25, random_state=0)
	classifier_conf = SVC(kernel='poly',C = 1.0, gamma = 'auto', probability=True)
	classifier_conf.fit(X, y)

	print (classifier_conf.score(X, y))
	#plotData (X, y)
	saveModel (classifier_conf)
	return classifier_conf
	#return classifier_conf

def readImageAndTrain () :
	X = []
	Y = []
	'''
	import os.path
	if os.path.isfile('svmModel.pkl') :
		with open('svmModel.pkl', 'rb') as fid:
			gnb_loaded = cPickle.load(fid)
		return gnb_loaded
	'''
	for folderName in range(2, 10) :
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
				x, y = getHistFromImage(name, img_name)
				X.extend(x)
				Y.extend(y)
				print ('Done folder', name, 'image number', img_name)
	#print (X)
	#print(Y)
	# print (X)
	# print (Y)
	return trainingSVM(X,Y);
	#trainingSVM (X, Y)



def diff(img,img1): # returns just the difference of the two images
      return cv2.absdiff(img,img1)
    
def remove_bg(img0,img,img1): # removes the background but requires three images 
        x = diff(img0,img)
        y = diff(img,img1)
        return cv2.bitwise_and(x,y)

def subtractBackground (img, img0) :

	return remove_bg(img, img0, img);
 
# cv2.imshow('final',d)

# cv2.waitKey(0)


def getHistFromImage (folderName, img_no):
	img = cv2.imread('../../'+ folderName + '/pcd' + img_no + 'r.png')
	# img_bg = cv2.imread('../../backgrounds/pcdb0003r.png')
	# img = subtractBackground (img_bg, img);
	#print(os.path.abspath('../../'+ folderName + '/pcd' + img_no + 'r.png'))
	# bw_img_bg = cv2.imread('../../backgrounds/pcdb0003r.png', 0)
	# bw_img = cv2.imread('../../'+ folderName + '/pcd' + img_no + 'r.png', 0)
	# #bw_img_bg = cv2.imread('../../'+ folderName + '/pcd' + img_no + 'r.png', 0)
	# bw_img = subtractBackground (bw_img_bg, bw_img);

	# plt.subplot (121), plt.imshow (img)
	# plt.subplot (122), plt.imshow (bw_img)
	# plt.show()

	sobelEdge = sobelEdgeDetection(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
	cannyEdge = cannyEdgeDetection(copy.deepcopy(img))
	lawsMasks = applyLawsMask(copy.deepcopy(img))


	lawsMasks.append(sobelEdge)
	lawsMasks.append(cannyEdge)
	fcs = focusImg(img)
	fcs_hist = build_histogram(fcs, 5)
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
		all_hists.extend(fcs_hist)
		# for j in range (11) :
		# 	for i in range(len(aRectangle) * int(255/bin_size) + 1):
		# 		# bins_arr.append(0)
		# 		print(i)
		# 		bc.append(i * bin_size)

		for features in aRectangle:
			# print(features.max())
			features = np.matmul(features, features.transpose())
			# print(features.max())
			if (features.max() != 0):
				features *= int(255.0/float(features.max()))

			# print(features.max())

			rect_hist = build_histogram(features, 5)
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
		all_hists.extend(fcs_hist)

		# for j in range (11) :
		# 	for i in range(len(aRectangle) * int(255/bin_size) + 1):
		# 		# bins_arr.append(0)
		# 		print(i)
		# 		bc.append(i * bin_size)

		for features in aRectangle:
			# print(features)
			features = np.matmul(features, features.transpose())

			if (features.max() != 0):
				features *= int(255.0/float(features.max()))
			rect_hist = build_histogram(features, 5)
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

