import numpy as np
import cv2


def combineImages(img1,img2):
	size = len(img1)

	newImage = np.zeros((size, size))

	for x in range(size):
		for y in range(size):
			newImage[x][y] = min(img1[x][y],img2[x][y])

	return newImage

sigmas = range(0,10)
print sigmas
lambdas = range(5,15)
print lambdas
gammas = [x / 10.0 for x in range(0, 10, 1)]
print gammas
psis = [x / 10.0 for x in range(0, 10, 1)]
print psis

for s in sigmas:
	print s
	for l in lambdas:
		for g in gammas:
			for p in psis:
				g_kernel = cv2.getGaborKernel((21, 21), s, np.pi/2, l, g, p, ktype=cv2.CV_64F)
				img = cv2.imread('pineapple.jpg')
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
				cv2.imwrite('pineappleTest/s'+str(s)+"l"+str(l)+"g"+str(g)+"p"+str(p)+'.jpg', filtered_img)




# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

g_kernel = cv2.getGaborKernel((21, 21), 4.0, np.pi/2, 10.0, 0.5, 0, ktype=cv2.CV_64F)

img1 = cv2.imread('pineapple.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
filtered_img1 = cv2.filter2D(img1, cv2.CV_8UC3, g_kernel)

#cv2.imwrite('origImage.jpg', img1)
cv2.imwrite('filteredImage.jpg', filtered_img1)

# h, w = g_kernel.shape[:2]
# g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('gabor kernel (resized)', g_kernel)



g_kernel = cv2.getGaborKernel((21, 21), 4.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_64F)

img2 = cv2.imread('pineapple.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
filtered_img2 = cv2.filter2D(img2, cv2.CV_8UC3, g_kernel)

#cv2.imshow('image', img1)
cv2.imwrite('filteredImage-2.jpg', filtered_img2)

combImg = combineImages(filtered_img1,filtered_img2)

cv2.imwrite('combinedImage.jpg', combImg)

cv2.waitKey(0)
cv2.destroyAllWindows()