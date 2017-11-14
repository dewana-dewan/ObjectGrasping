import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# 0. Read the image
image  = scipy.misc.imread('./samples/pcd0312r.png',mode="L")
img_arr = np.array(image)
print (image)
# 1. Get the histogram of the iamge
cn = []

for i in range(256) :
    x = float(i+1)/2
    cn.append(x)

print (len(cn))
print (img_arr.shape[0])
print(img_arr.shape[1])

for i in range (img_arr.shape[0]) :
    for j in range (img_arr.shape[1]) :
        img_arr[i][j] = float(img_arr[i][j])

bn = []

for i in range(256):
    bn.append(float(i))

hist, bin_edges = np.histogram(img_arr, bins=cn)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

print (hist)

import matplotlib.pyplot as plt
plt.plot(bin_centers, hist, lw=2)
plt.show()
