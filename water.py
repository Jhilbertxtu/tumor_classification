import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img/test22.jpg',0)
img1=cv2.imread('img/test22.jpg')

ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
   
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

ret,markers = cv2.connectedComponents(sure_fg)

markers = markers+1
 
markers[unknown==255] = 0

markers = cv2.watershed(img1,markers)
img1[markers == -1] = [255,0,0]

#markers.show()
cv2.imwrite('img/water11.jpg', img1)