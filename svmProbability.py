import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
plt.rcParams['figure.figsize'] = (12, 6)
style.use('ggplot')


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
                        alpha=0.4, linestyles=['--', '-', '--'])
    ax.scatter(clf.support_vectors_[:, 0], 
                clf.support_vectors_[:, 1], 
                s=100, linewidth=1, facecolors='none')
    plt.show()
    return



X1, Y1 = make_classification(n_samples = 700, n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(X1)
data = X1
#plt.scatter(data[:, 0], data[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# X = np.array([[.2, .1], [.6, .2], [.5, .3], [.3, .0], [5.0, .4], [.1, .1]])
# y = np.array([0, 1, 1, 0, 1, 0])

# classifier_conf = SVC(kernel='linear',C = 1000, gamma = 'auto', probability=True)
# classifier_conf.fit(X, y)
# pro1 = classifier_conf.predict_proba(np.array([[.1, .3]]))
# pro2 = classifier_conf.predict_proba(np.array([[.4,.2]]))

# print (pro1);
# print (pro2);

#X, y = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=2, cov=3)

draw_svm (data, Y1)

#plt.show()



