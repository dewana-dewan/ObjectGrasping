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
			# font = cv2.FONT_HERSHEY_SIMPLEX
			# cv2.putText(img, str(cnt),pt , font, .5, (50,0,0),2,cv2.LINE_AA)
			# print(pt, pts)

			if cnt % 4 == 0:
				# print(cnt, pts)
				# rectangles.append(convert_to_matrix(img, pts))
				print(pts)
				x_c = (pts[0] + pts[4])/2
				y_c = (pts[1] + pts[5])/2
				width = int(((pts[0] - pts[2]) ** 2 + (pts[1] - pts[3]) ** 2) ** (1/2))
				height = int(((pts[2] - pts[4]) ** 2 + (pts[3] - pts[5]) ** 2) ** (1/2))
				if ((pts[0] - pts[2]) != 0):
					angl = atan(pts[1] - pts[3]) / (pts[0] - pts[2])
				else:
					angl = np.pi / 2
				print(width, height, angl, x_c, y_c)

				rect = subimage(img, (110, 125), np.pi / 6.0, 100, 200)
				plt.subplot(111),plt.imshow(rect)
				plt.title('Positive rectangles'), plt.xticks([]), plt.yticks([])
				plt.show()
				pts = pts.reshape((-1,1,2))
				# print(cnt, pts)
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				print(pts[0][0])
				pts = np.array([], np.int32)
			cnt += 1

	return img

def subimage(image, center, theta, width, height):
    # theta *= 3.14159 / 180 # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)



# def subimage(image, centre, theta, width, height):
#    output_image = cv2.cv.CreateImage((width, height), image.depth, image.nChannels)
#    mapping = np.array([[np.cos(theta), -np.sin(theta), centre[0]],
#                        [np.sin(theta), np.cos(theta), centre[1]]])
#    map_matrix_cv = cv2.fromarray(mapping)
#    cv2.GetQuadrangleSubPix(image, output_image, map_matrix_cv)
#    return output_image

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
			if (m >= 1):
				for i in range (cord[0], cord[2]):
					print(bresenham())
	

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
