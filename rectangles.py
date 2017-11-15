import cv2
import copy
import time
from math import *
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

			if cnt % 4 == 0:				
				print(pts)
				if ((pts[2] - pts[0]) != 0):
					m = (pts[3] - pts[1]) / (pts[2] - pts[0])
					angl = atan(m)*180/(np.pi)
				else:
					angl = 90
				c_x = int((pts[0] + pts[4])/2)
				c_y = int((pts[1] + pts[5])/2)
				wt = int(((pts[3] - pts[1])**2 + (pts[2] - pts[0])**2)**(1/2))
				ht = int(((pts[7] - pts[1])**2 + (pts[6] - pts[0])**2)**(1/2))
				
				print(atan(m), angl, c_x, c_y, ht, wt)
				cv2.circle(img, (c_x, c_y) , 1, (50,0,0))

				rect = subimage(img, (c_x, c_y), atan(m)*180/(np.pi), wt, ht)
				rectangles.append(rect)
				
				### transpose
				new = np.matmul(rect, rect.transpose())
				### ding
				
				plt.subplot(111),plt.imshow(new)
				plt.title('Rectangles'), plt.xticks([]), plt.yticks([])
				plt.show()
				
				pts = pts.reshape((-1,1,2))
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				print(pts[0][0])
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
