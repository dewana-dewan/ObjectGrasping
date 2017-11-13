import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./samples/pcd0312r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

edges = cv2.Canny(img,50,100)
bw_edges = cv2.Canny(gray,50,100)
blur_edges = cv2.Canny(blurred,50,100)

kernel = np.ones((3, 3), np.uint8)
margin = 30
dilated = cv2.morphologyEx(bw_edges, cv2.MORPH_DILATE, kernel)

with_contours, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

max1 = 0
max1_l = 0
max2 = 0
max2_l = 0
for i in range(len(contours)):
	print(i, len(contours[i]), max1, max2)
	if max1_l <= len(contours[i]):
		max1_l = len(contours[i])
		max2_l = max1_l
		max2 = max1
		max1 = i
	elif max2_l <= len(contours[i]):
		max2_l = len(contours[i])
		max2 = i

print(max2, max1)

x, y, width, height = cv2.boundingRect(contours[max2])
roi = img[y - margin:y + height + margin, x - margin:x + width + margin]
# print(contours)
# cv2.drawContours(with_contours, contours, -1, (0,255,0), 3)

plt.subplot(131),plt.imshow(blur_edges)
plt.title('Blurred Image edges'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(dilated)
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(roi)
plt.title('BW Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
