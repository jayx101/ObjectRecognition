import cv2
import numpy as np
import imagetransform as it
from collections import deque

def removeBackGround(im):
    '''Remove BG and find the contours
    Params:
        img: image to process
    Returns:
        threshold: image to see what image is being processed by system
    '''
    fgmask = fgbg.apply(im)

    fgmask = cv2.GaussianBlur(fgmask, (9, 9), 0)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)

    # im_test = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(fgmask, 200, 255, 0)

    return thresh

def getContours(thresh, drawObj=True):
    radius = 0
    # get countours from thresholded image
    _, cnts, hierarchy = cv2.findContours(thresh.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            pass

        # only proceed if the radius meets a minimum size
        if radius > 10 and drawObj is True:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(im_detect, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(im_detect, center, 5, (0, 0, 255), -1)
        if radius > 10:
            pts.appendleft(center)

    return pts, cnts, radius

def createDetectBox(img, screen_quadrants=3):
    im_rect = img.copy()
    img_height, img_width, _ = im_rect.shape

    detect_x = (img_width / screen_quadrants) * (screen_quadrants - 1)

    cv2.rectangle(im_rect, (detect_x, 0), (img_width,
                           img_height), (0, 255, 0), 3)

    return im_rect, detect_x

def montionDetect(pts, im_detect):
    '''
    '''
    (dX, dY, x, avgX) = (0, 0, 0, 0)
    direction = ""
    # loop over the set of tracked points
    for i in np.arange(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        avgX = (pts[0][0] - pts[len(pts) - 1][0]) / len(pts)

        # check to see if enough points have been accumulated in
        # the buffer
        if counter >= 10 and i == 10 and pts[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            x = pts[0][0]
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                dirX = "Right" if np.sign(dX) == 1 else "Left"

            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                dirY = "Up" if np.sign(dY) == 1 else "Down"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = "{}-{}".format(dirY, dirX)

            # otherwise, only one direction is non-empty
            else:
                direction = dirX if dirX != "" else dirY

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(buff / float(i + 1)) * 2.5)
        cv2.line(im_detect, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the movement deltas and the direction of movement on
    # the frame
    cv2.putText(im_detect, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 3)
    cv2.putText(im_detect, "dx: {}, dy: {}, x {}".format(dX, dY, x),
        (10, im_detect.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1)

    return im_detect, (dX, dY, x, avgX)


cap = cv2.VideoCapture(1)
fgbg = cv2.createBackgroundSubtractorMOG2(500, -1, True)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
buff = 30
pts = deque(maxlen=buff)
counter = 0
objFound = False

while(1):
    _, im = cap.read()
    im = it.resize(im, 600)

    img_height, img_width, _ = im.shape

    # draw box where we want to detect the oject
    im_detect, detect_x = createDetectBox(im, 2)
    # remove the background, find only moving part
    thresh = removeBackGround(im)
    # get contours
    pts, cnts, radius = getContours(thresh, objFound is False)
    img = cv2.drawContours(im.copy(), cnts, -1, (0, 255, 0), 3)

    im_detect, movement = montionDetect(pts, im_detect)

    if objFound is False and abs(movement[0]) > 20 and movement[2] > detect_x and radius > 10:
        objFound = True
        cv2.imwrite("foundimg.jpg", im) # writes image test.bmp to disk
    elif abs(movement[0]) > 20 and movement[2] < detect_x and radius > 10:
        objFound = False
    elif objFound is True:
        cv2.putText(im_detect, "Recorded new object!",
            (230, im_detect.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)

    cv2.imshow('frame', im)
    cv2.imshow('frame1', thresh)
    cv2.imshow('frame2', im_detect)

    counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
