import scipy
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

image  = scipy.misc.imread('./BTOW.jpg',mode="L")
img_arr = np.array(image)
print (image)
# 1. Get the histogram of the iamge
# cn = []

# for i in range(256) :
#     x = float(i+1)/2
#     cn.append(x)

# print (len(cn))
# print (img_arr.shape[0], img_arr.shape[1])

# for i in range (img_arr.shape[0]) :
#     for j in range (img_arr.shape[1]) :
#         img_arr[i][j] = float(img_arr[i][j])

# bn = []

# for i in range(256):
#     bn.append(float(i))

# hist, bin_edges = np.histogram(img_arr, bins=cn)
# bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

bins_arr = []
bins_arr1 = []
bc = []
bin_size = 10

for i in range(int(255/bin_size) + 1):
	print(i)
	bins_arr.append(0)
	bins_arr1.append(0)

	bc.append(i * bin_size)

print(bins_arr, len(bins_arr))

for i in range(img_arr.shape[0]):
	for j in range(img_arr.shape[1]):
		
		a = int(img_arr[i][j] / bin_size)
		b = img_arr[i][j] % bin_size
		k = abs(0.5 - b / bin_size)
		# print(img_arr[i][j] , a, b, k)
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

for i in range(img_arr.shape[0]):
	for j in range(img_arr.shape[1]):
		
		a = int(img_arr[i][j] / bin_size)
		bins_arr1[a] += 1

# for i in range(int(255/bin_size) + 1):
# 	c += bins_arr[i]


print(bins_arr, c, len(bins_arr))
# print (hist)

# 3. Plot the histogram
# import matplotlib.pyplot as plt
#plt.plot(bin_centers, hist, lw=2)

plt.subplot(121),plt.plot(bc, bins_arr, lw=2)
plt.subplot(122),plt.plot(bc, bins_arr1, lw=2)
plt.show()
