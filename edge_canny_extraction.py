import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dewan.JPG')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

edges = cv2.Canny(img,50,100)
bw_edges = cv2.Canny(gray,50,100)
blur_edges = cv2.Canny(blurred,50,100)

plt.subplot(131),plt.imshow(blur_edges)
plt.title('Blurred Image edges'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(bw_edges)
plt.title('BW Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
