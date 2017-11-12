import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('./samples/pcd0312r.png',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
dst = cv2.warpAffine(img,M,(cols,rows))

plt.imshow(dst,cmap = 'gray')
plt.show()
