import cv2
import numpy as np
from math import *
from matplotlib import pyplot as plt

def subimage(image, center, theta, width, height):
    theta *= 3.14159 / 180 # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

img = cv2.imread('./samples/pcd0312r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img,50,100)
bw_edges = cv2.Canny(gray,50,100)

kernel = np.ones((3, 3), np.uint8)
margin = 20
dilated = cv2.morphologyEx(bw_edges, cv2.MORPH_DILATE, kernel)

with_contours, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

max1 = 0
max1_l = 0
max2 = 0
max2_l = 0
for i in range(len(contours)):
	print(i, len(contours[i]), max1, max2)
	# print()
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

# roi containes the focused image now.
# we'll now try and create rectangles

print(roi.shape[0], roi.shape[1])

siz_x = int(roi.shape[0] / 5)
siz_y = int(roi.shape[1] / 5)
all_for_test = []

for i in range(0, roi.shape[0] - siz_x, 10):
	for j in range(0, roi.shape[1] - siz_y, 10):

		local_roi = roi[j:j + siz_y, i:i + siz_x]
		bw_local_roi = cv2.cvtColor(local_roi,cv2.COLOR_BGR2GRAY)
		
		ret, thresh = cv2.threshold(bw_local_roi ,127,255,0)
		im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


		if ( len(contours) >= 1 and len(contours[0]) <= 4 ):
			print(len(contours[0]))
			continue;
		
		print(len(contours[0]))

		cv2.rectangle(roi,(i, j), (i + siz_x, j + siz_y), (0,255,0),1)

		center = (i + siz_x/2, j + siz_y/2)
		theta = -1 * 45
		dings = []
		for k in range(3):
			ding = subimage(roi, center, theta, siz_x, siz_y)
			dings.append(ding)
			theta += 45

		# plt.subplot(231),plt.imshow(roi)
		# plt.subplot(232),plt.imshow(local_roi)		
		# plt.subplot(233),plt.imshow(thresh)

		# plt.subplot(234),plt.imshow(dings[0])
		# plt.subplot(235),plt.imshow(dings[1])		
		# plt.subplot(236),plt.imshow(dings[2])				
		# plt.show()
		all_for_test.extend(dings)
