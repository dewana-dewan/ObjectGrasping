import cv2
import copy
import time
import numpy as np
from matplotlib import pyplot as plt

def plot_rectangles(file_name, img):

	pts = np.array([], np.int32)
	cnt = 1
	rectangles = []

	with open(file_name, 'r') as fl:
		for line in fl:
	
			content = line.strip().split(' ')
			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)
			# print(pt, pts)

			if cnt % 4 == 0:
				# print(cnt, pts)
				rectangles.append(convert_to_matrix(img, pts))
				pts = pts.reshape((-1,1,2))
				# print(cnt, pts)
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				pts = np.array([], np.int32)
			cnt += 1

	return img

def convert_to_matrix(img, cord):
	if (cord[1] == cord[3]):
		rect = img[cord[5]:cord[1], cord[2]:cord[0]]
		print(cord, rect)
		print((cord[1], cord[3]), (cord[0], cord[2]))
		plt.subplot(111),plt.imshow(rect)
		plt.title('Positive rectangles'), plt.xticks([]), plt.yticks([])
		plt.show()
		return rect
	else:
		if (cord[2] - cord[0]) == 0:
			pass
		else:
			m = (cord[1] - cord[3]) / (cord[0] - cord[2])
			print((cord[1], cord[3]), (cord[0], cord[2]), m)
	

img = cv2.imread('./samples/pcd0312r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

bw_edges = cv2.Canny(gray,50,100)

pos_rect = plot_rectangles('./samples/pcd0312cpos.txt', copy.copy(bw_edges))
neg_rect = plot_rectangles('./samples/pcd0312cneg.txt', copy.copy(bw_edges))

plt.subplot(121),plt.imshow(pos_rect)
plt.title('Positive rectangles'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(neg_rect)
plt.title('Negative Rectangles'), plt.xticks([]), plt.yticks([])

plt.show()