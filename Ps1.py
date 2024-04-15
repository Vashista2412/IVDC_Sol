import cv2
import numpy as np

image = cv2.imread('street_image.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[-1, -1, -1],
                   [ 2,  2,  2],
                   [-1, -1, -1]])

filtered_image = cv2.filter2D(gray_image, -1, kernel)

cv2.imwrite('filtered_street_image.jpg', filtered_image)

cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
