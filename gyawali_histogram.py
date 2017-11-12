import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# 0. Read the image
image  = scipy.misc.imread('samples/pcd0312r.png',mode="L")
img_arr = np.array(image)
print (image)
# 1. Get the histogram of the iamge
cn = []

for i in range(256) :
    x = float(i+1)/2
    cn.append(x)

print (len(cn))
print (img_arr.shape[0], img_arr.shape[0])

for i in range (img_arr.shape[0]) :
    for j in range (img_arr.shape[1]) :
        img_arr[i][j] = float(img_arr[i][j])

bn = []

for i in range(256):
    bn.append(float(i))

hist, bin_edges = np.histogram(img_arr, bins=cn)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

print(bin_centers)

bins_arr = []
bc = []
bin_size = 30

for i in range(int(255/bin_size) + 1):
	print(i)
	bins_arr.append(0)
	bc.append(i * bin_size)

print(bins_arr)

for i in range(img_arr.shape[0]):
	for j in range(img_arr.shape[1]):
		
		a = int(img_arr[i][j] / bin_size)
		b = img_arr[i][j] % bin_size
		k = abs(0.5 - b / bin_size)
		# round(k, 3)
		# print(k)

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

print(bins_arr, c)

# print (hist)

# 3. Plot the histogram
# import matplotlib.pyplot as plt
#plt.plot(bin_centers, hist, lw=2)

plt.plot(bc, bins_arr, lw=2)
plt.show()