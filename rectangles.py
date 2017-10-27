import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('pcd0312r.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

bw_edges = cv2.Canny(gray,50,100)

with open('pcd0312cpos.txt', 'r') as fl:
	for line in fl:
		content = fl.readline().strip().split(' ')
		print(content)

fl.close()
cv2.rectangle(bw_edges, (20,20), (100,100), (255,0,0), 1)

plt.subplot(111),plt.imshow(bw_edges)
plt.title('Blurred Image edges'), plt.xticks([]), plt.yticks([])

plt.show()
