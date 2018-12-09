import cv2
import numpy as np
import scipy as ndimage


#to read rgb image
vehicle_rgb=cv2.imread('image.jpg')

#to convert rgb to grayscale
vehicle_gray=cv2.imread('image.jpg',cv2.IMREAD_GRAYSCALE)

#to convert b/w to binary image using otsu's threshold method
#(thresh, vehicle_binary) = cv2.threshold(vehicle_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


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


##################################################high pass filter 5x5

# data = np.asarray(vehicle_floodfill_inv)
# data1 = data.flatten()
# kernel = np.array([[-1, -1, -1, -1, -1],
#                    [-1,  1,  2,  1, -1],
#                    [-1,  2,  4,  2, -1],
#                    [-1,  1,  2,  1, -1],
#                    [-1, -1, -1, -1, -1]])
# highpass_5x5 = ndimage.convolve(data1, kernel)

####################################################

####################################################

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(vehicle_floodfill)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(vehicle_floodfill, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


####################################################




#to detect edges using canny edge detection
vehicle_canny=cv2.Canny(vehicle_binary,100,200)




#Image Opening, morphological processing to perform contour filling
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# vehicle_filling = cv2.morphologyEx(vehicle_gray,cv2.MORPH_OPEN,kernel)


#To pad the canny image with a black border
# row, col= vehicle_canny.shape[:2]
# bottom= vehicle_canny[row-2:row, 0:col]
# mean= cv2.mean(bottom)[0]
#
# bordersize=10
# border=cv2.copyMakeBorder(vehicle_canny, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )






# #To fill the holes
# gray = cv2.cvtColor(vehicle_rgb, cv2.COLOR_BGR2GRAY)
#
# threshold1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,3,1)
#
# _, contours, hierarchy = cv2.findContours(threshold1,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
#
# mask = np.zeros(vehicle_rgb.shape[:-1],np.uint8)
#
# cv2.drawContours(mask,contours,-1,(255,255,255),-1)
#
# height, width = vehicle_rgb.shape[:-1]
#
# mask1 = np.zeros((height+2, width+2), np.uint8)
# cv2.floodFill(mask,mask1,(0,0),255)
# mask_inv=cv2.bitwise_not(mask)







#to write the b/w image
cv2.imwrite('image_bw.jpg',vehicle_gray)

#to write the b/w image
cv2.imwrite('image_binary.jpg',vehicle_binary)

#to write the edge detected image
cv2.imwrite('image_canny.jpg',vehicle_canny)

#to write the hole filled image
cv2.imwrite('image_filled.jpg',vehicle_gray)

#to write the border filled image
# cv2.imwrite('image_border.jpg',border)

#to write the hole filled image
# cv2.imwrite('image_holefilled.jpg',mask_inv)

#to write the hole filled image
cv2.imwrite('image_holefilled.jpg',vehicle_floodfill)

#to write the detected blobs image
cv2.imwrite('image_final.jpg',im_with_keypoints)

#to write the hole filled image
# cv2.imwrite('image_highpass.jpg',highpass_5x5)
