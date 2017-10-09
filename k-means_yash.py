#!/usr/bin/python
from PIL import Image, ImageStat, ImageFilter
import numpy
import cv2
from matplotlib import pyplot as plt


# Will detewrmine if the centroids have converged or not.
# Essentially, if the current centroids and the old centroids
# are virtually the same, then there is convergence.
# Absolute convergence may not be reached, due to oscillating
# centroids. So a given range has been implemented to observe
# if the comparisons are within a certain ballpark
def converged(centroids, old_centroids):
	if len(old_centroids) == 0:
		return False


	if len(centroids) <= 5:
		a = 1
	elif len(centroids) <= 10:
		a = 2
	else:
		a = 4

	for i in range(0, len(centroids)):
		cent = centroids[i]
		old_cent = old_centroids[i]

		if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
			continue
		else:
			return False

	return True
#end converged



# Method used to find the closest centroid to the given pixel.
def getMin(pixel, centroids):
	minDist = 9999
	minIndex = 0

	for i in range(0, len(centroids)):
		d = numpy.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
		if d < minDist:
			minDist = d
			minIndex = i

	return minIndex
#end getMin



# Assigns each pixel to the given centroids for the algorithm.
# Method finds the closest centroid to the given pixel, then
# assigns that centroids to the pixel.
def assignPixels(centroids):
	clusters = {}

	for x in range(0, img_width):
		for y in range(0, img_height):
			p = px[x, y]
			minIndex = getMin(px[x, y], centroids)

			try:
				clusters[minIndex].append(p)
			except KeyError:
				clusters[minIndex] = [p]

	return clusters
#end assignPixels


# Method is used to  re-center the centroids according
# to the pixels assigned to each. A mean average is 
# applied to each cluster's RGB values, which is then
# set as the new centroids.
def adjustCentroids(centroids, clusters):
	new_centroids = []
	keys = sorted(clusters.keys())
	#print(keys)

	for k in keys:
		n = numpy.mean(clusters[k], axis=0)
		new = (int(n[0]), int(n[1]), int(n[2]))
		#print(str(k) + ": " + str(new))
		new_centroids.append(new)

	return new_centroids
#end adjustCentroids


# Used to initialize the k-means clustering
def startKmeans(someK):
	centroids = []
	old_centroids = []
	rgb_range = ImageStat.Stat(im).extrema
	i = 1

	#Initializes someK number of centroids for the clustering
	for k in range(0, someK):

		cent = px[numpy.random.randint(0, img_width), numpy.random.randint(0, img_height)]
		centroids.append(cent)
	

	print("Start of K-means")
	
	while not converged(centroids, old_centroids) and i <= 20:
		#print("Iteration #" + str(i))
		i += 1

		old_centroids = centroids 								#Make the current centroids into the old centroids
		clusters = assignPixels(centroids) 						#Assign each pixel in the image to their respective centroids
		centroids = adjustCentroids(old_centroids, clusters) 	#Adjust the centroids to the center of their assigned pixels


	print("Convergence Reached!")
	print(centroids)
	return centroids
#end startKmeans


# Once the k-means clustering is finished, this method
# generates the segmented image and opens it.
def drawWindow(result):
	img = Image.new('RGB', (img_width, img_height), "white")
	p = img.load()
	
	for x in range(img.size[0]):
		for y in range(img.size[1]):
			RGB_value = result[getMin(px[x, y], result)]
			p[x, y] = RGB_value

	#img.show()
	return img
#end drawWindow



num_input = str(input("Enter image number: "))
k_input = int(input("Enter K value: "))

#Input image
img = "img/test" + num_input.zfill(2) + ".jpg"
im = Image.open(img)
im.show()

img_width, img_height = im.size
im = im.filter(ImageFilter.UnsharpMask(3,150,3))

px = im.load()
result_Kmeans = startKmeans(k_input)
clustered_image = drawWindow(result_Kmeans)
clustered_image.show()

#PIL object to numpy array conversion
clustered_image = numpy.asarray(clustered_image)
r, g, b = cv2.split(clustered_image)
clustered_image = cv2.merge([b, g, r])
clustered_image = cv2.cvtColor(clustered_image,cv2.COLOR_BGR2GRAY)
input = clustered_image
#Thresholding and smoothing filters
ret2,clustered_image = cv2.threshold(clustered_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#kernel = numpy.ones((7,7),numpy.uint8)
#clustered_image = cv2.morphologyEx(clustered_image, cv2.MORPH_OPEN, kernel)

clustered_image = cv2.GaussianBlur(clustered_image,(3,3),0)
clustered_image = cv2.medianBlur(clustered_image,5)
ret3,clustered_image = cv2.threshold(clustered_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
clustered_image = cv2.medianBlur(clustered_image,7)
cv2.imwrite('img/test22.jpg', clustered_image)


im,contours,hierarchy = cv2.findContours(clustered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#im,contours,hierarchy = cv2.findContours(clustered_image, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print(len(contours))
largest_area = 0

for x in range(len(contours)):
	a = cv2.contourArea( contours[x]);
	if(a>largest_area):
		largest_area = a
		largest_contour_index = x
		
cv2.drawContours(clustered_image,contours,largest_contour_index,255,-1)
ret3,clustered_image = cv2.threshold(clustered_image,220,255,cv2.THRESH_BINARY)
#clustered_image1 = 255 - clustered_image
#clustered_image1 = clustered_image1 - input
cv2.imwrite('img/test26.jpg', clustered_image)

'''

if len(contours) != 0:
    
    #cv2.drawContours(clustered_image, contours, -1, 255, 3)
    c,i = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    tumor = cv2.rectangle(clustered_image,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.drawContours(clustered_image, contours, -1, 255, 3)

tumor_seed_point = (x+w/2,y+h/2)
print(tumor_seed_point)

	

cv2.imwrite('img/test25.jpg', clustered_image)
# show the images
cv2.imshow("Result", numpy.hstack([input, clustered_image]))

'''