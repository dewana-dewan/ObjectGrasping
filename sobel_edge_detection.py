import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('samples/pcd0312r.png',0)

kernel = np.ones((5,5),np.float32)/25
img = cv2.filter2D(img,-1,kernel)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(sobely + sobelx ,cmap = 'binary')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'binary')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'binary')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()
