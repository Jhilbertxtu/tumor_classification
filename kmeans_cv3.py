from PIL import Image, ImageStat, ImageFilter
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img/test01.jpg')
Z = np.asarray(img,dtype="float32")

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS , 1000, 20.0)
print(criteria)
K = 20
ret,label,center=cv2.kmeans(Z,K,criteria,5000,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

#cv2.imshow('res2',res2)
cv2.imwrite('img/cv3.jpg',res2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()