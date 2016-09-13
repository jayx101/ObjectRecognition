# Standard imports
import cv2
import numpy as np
import imagetransform as it

# Read image
im = cv2.imread("blob_edit.jpg", cv2.IMREAD_GRAYSCALE)

im = it.resize(im)  
im = cv2.GaussianBlur(im, (9, 9), 0)

# it.display(im) 
# Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector()
 

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 256
 
# Filter by Area.
# pakams.filterByArea = True
# params.minArea =800
params.filterByColor = True 
params.blobColor = 255  
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0
 
# Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.5
 
# Filter by Inertia
# params.filterByInertia =True
# params.minInertiaRatio =0.1
 
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
# reversemask=255-mask
# keypoints = detector.detect(reversemask)


# Detect blobs.
keypoints = detector.detect(im)
 
print keypoints
print len(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
