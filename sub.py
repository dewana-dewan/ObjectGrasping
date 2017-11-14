import cv2
import numpy as np
def diff(img,img1): # returns just the difference of the two images
      return cv2.absdiff(img,img1)
    
def remove_bg(img0,img,img1): # removes the background but requires three images 
        x = diff(img0,img)
        y = diff(img,img1)
        return cv2.bitwise_and(x,y)

a = cv2.imread('abc.png',0)
b = cv2.imread('xyz.png',0)
c = cv2.imread('abc.png',0)

d = remove_bg(a,b,c)
 
cv2.imshow('final',d)

cv2.waitKey(0)
cv2.destroyAllWindows()


