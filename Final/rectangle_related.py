import numpy as np
from math import *
import cv2
from matplotlib import pyplot as plt

# class Rectangle_related():
def plot_rectangles(file_name, img):

	pts = np.array([], np.int32)
	cnt = 1
	rectangles = []

	with open(file_name, 'r') as fl:
		for line in fl:

			content = line.strip().split(' ')
			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)

			if cnt % 4 == 0:
				# print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))

				# print(atan(m), angl, c_x, c_y, ht, wt)
				cv2.circle(img, (c_x, c_y) , 1, (50,0,0))

				rect = subimage(img, (c_x, c_y), atan(m)*180/(np.pi), wt, ht)
				rectangles.append(rect)

				plt.subplot(111),plt.imshow(rect)
				plt.title('Rectangles'), plt.xticks([]), plt.yticks([])
				plt.show()

				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				# print(pts[0][0])
				pts = np.array([], np.int32)
			cnt += 1

	return img


def subimage(image, center, theta, width, height):
	theta *= 3.14159 / 180 # convert to rad

	v_x = (cos(theta), sin(theta))
	v_y = (-sin(theta), cos(theta))
	s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
	s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

	mapping = np.array([[v_x[0],v_y[0], s_x],
						[v_x[1],v_y[1], s_y]])

	return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)

def getRectangles(file_name, lawsMasks):

	pts = np.array([], np.int32)
	cnt = 0
	rectangles = []
	k = 1;
	flag = False
	skip = 0;
	with open(file_name, 'r') as fl:
		for line in fl:
			cnt += 1
			this_rectangle = []
			content = line.strip().split(' ')

			if (flag and k == skip) :
				flag = False
				k = 1
				skip = 0
				continue
			if flag :
				k += 1
				continue

			if content[0] == 'NaN' or content[1] == 'NaN' :
				k = 1
				pts = np.array([], np.int32)
				if (cnt % 4 == 0) :
					continue
				else :
					skip = 4 - (cnt%4)
					flag = True
					continue

			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)
			#print(pts)

			if cnt % 4 == 0:
				# print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))

				#print(atan(m), angl, c_x, c_y, ht, wt)
				pts = np.array([], np.int32)

				k = 0

				for img in lawsMasks:
					rect = subimage(img, (c_x, c_y), angl, wt, ht)
					# plt.subplot(3,4,k+1),plt.imshow(rect)
					# k += 1
					# # plt.subplot(111),plt.imshow(rect)
					# plt.title(str(k)), plt.xticks([]), plt.yticks([])
					this_rectangle.append(rect)

				plt.show()
				rectangles.append(this_rectangle)

				# cv2.circle(img, (c_x, c_y) , 1, (50,0,0))
				# pts = pts.reshape((-1,1,2))
				# cv2.polylines(img, [pts], True, (100,0,0), 1)
				# print(pts[0][0])
				# pts = np.array([], np.int32)

	return rectangles

def createRectangles(img):
	# print (img.shape)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print (gray.shape)
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
		# print(i, len(contours[i]), max1, max2)
		# print()
		if max1_l <= len(contours[i]):
			max2_l = max1_l
			max1_l = len(contours[i])
			max2 = max1
			max1 = i
		elif max2_l <= len(contours[i]):
			max2_l = len(contours[i])
			max2 = i

	# print(max2, max1)

	x, y, width, height = cv2.boundingRect(contours[max2])
	roi = img[y - margin:y + height + margin, x - margin:x + width + margin]

	# roi containes the focused image now.
	# we'll now try and create rectangles

	# print(roi.shape[0], roi.shape[1])

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

			if ( len(contours) == 0):
				continue

			#cv2.rectangle(roi,(i, j), (i + siz_x, j + siz_y), (0,255,0),1)

			center = (i + siz_x/2, j + siz_y/2)
			# theta = -1 * 45
			theta = 0
			dings = []
			# ding = np.array([], np.int32)
			print('ding')
			# for k in range(3):
			ding = subimage(roi, center, theta, siz_x, siz_y)
			print (ding.shape, '------')
			print('ding')
			# dings.append(ding)
			# print (len(dings), '*********')
			theta += 45
			# return ;
			# plt.subplot(231),plt.imshow(roi)
			# plt.subplot(232),plt.imshow(local_roi)
			# plt.subplot(233),plt.imshow(thresh)
			#
			# plt.subplot(234),plt.imshow(dings[0])
			# # plt.subplot(235),plt.imshow(dings[1])
			# # plt.subplot(236),plt.imshow(dings[2])
			# plt.show()
			# return dings
			print(len(ding), 'akjdkjhaskdh')
			all_for_test.append(ding)
			print('all test length', len(all_for_test))
	return all_for_test

