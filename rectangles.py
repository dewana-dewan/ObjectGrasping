import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_rectangles(file_name, img):

	pts = np.array([], np.int32)
	cnt = 1

	with open(file_name, 'r') as fl:
		for line in fl:
	
			content = line.strip().split(' ')
			pt = (int(float(content[0])), int(float(content[1])))
			pts = np.append(pts, pt)
			print(pt, pts)

			if cnt % 4 == 0:
				pts = pts.reshape((-1,1,2))
				print(cnt, pts)
				cv2.polylines(img, [pts], True, (100,0,0), 1)
				pts = np.array([], np.int32)
			cnt += 1
	

img = cv2.imread('pcd0312r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

bw_edges = cv2.Canny(gray,50,100)

plot_rectangles('pcd0312cpos.txt', bw_edges)
#plot_rectangles('pcd0312cneg.txt', bw_edges)

plt.subplot(111),plt.imshow(bw_edges)
plt.title('Blurred Image edges'), plt.xticks([]), plt.yticks([])

plt.show()

