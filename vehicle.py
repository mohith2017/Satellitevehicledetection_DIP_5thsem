import cv2
import numpy as np
import scipy as ndimage


#to read rgb image
vehicle_rgb=cv2.imread('image.jpg')

#to convert rgb to grayscale
vehicle_gray=cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)

#############################################################filling holes

th, vehicle_binary = cv2.threshold(vehicle_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU);

# Copy the thresholded image.
vehicle_floodfill = vehicle_binary.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = vehicle_binary.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(vehicle_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
vehicle_floodfill_inv = cv2.bitwise_not(vehicle_floodfill)

####################################################

####################################################Blob detection 

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Filter by Area.
params.filterByArea = True
params.minArea = 1500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(vehicle_floodfill)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(vehicle_floodfill, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
####################################################canny edge detection(optional)

#to detect edges using canny edge detection
vehicle_canny=cv2.Canny(vehicle_binary,100,200)

####################################################

###################################################step-by-step output images

#to write the b/w image
cv2.imwrite('image_bw.jpg',vehicle_gray)

#to write the b/w image
cv2.imwrite('image_binary.jpg',vehicle_binary)

#to write the edge detected image
cv2.imwrite('image_canny.jpg',vehicle_canny)

#to write the hole filled image
cv2.imwrite('image_filled.jpg',vehicle_gray)

#to write the hole filled image
cv2.imwrite('image_holefilled.jpg',vehicle_floodfill)

#to write the detected blobs image
cv2.imwrite('image_final.jpg',im_with_keypoints)

#####################################################

print("Number of cars are: ",number_of_blobs)
