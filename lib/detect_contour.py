import cv2
import random
import numpy as np
import imagetransform as it
from collections import deque
import classification as cs
from sklearn.svm import LinearSVC
# constants
RADIUS = 50
TRAIN_PATH = '../dataset/train'
TEST_PATH = '../dataset/test'

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)
# fgbg = cv2.createBackgroundSubtractorMOG2(500, -1, True)
fgbg = cv2.BackgroundSubtractorMOG2(300, 100, True)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
buff = 30
pts = deque(maxlen=buff)
counter = 0
y_pred = 0
objFound = False
a = 0

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

    # fgmask = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(fgmask, 200, 255, 0)

    return thresh

def getContours(thresh, drawObj=True):
    radius = 30
    # get countours from thresholded image
    cnts, hierarchy = cv2.findContours(thresh.copy(),
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
        if radius > RADIUS and drawObj is True:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(im_detect, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(im_detect, center, 5, (0, 0, 255), -1)
        if radius > RADIUS:
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

def predictImageAndStore(img, isdump=False):
    '''
    Predict a sinigle image
    Return predition, dist from hyperplane
    '''

    weights = cs.getClassWeights()
    train_data = cs.getTrainData(False)
    clf = LinearSVC(class_weight=weights)
    clf.fit(train_data[0], train_data[1])

    y_pred, dist_hyp = cs.predictImg(clf, img)
    path = cs.TRAIN_PATH + cs.class2Name(y_pred)[0] + '/'
    if isdump: path = '../dataset/dump/'    # if isdump flag is set just put result in dump folder
    # get the classname from code
    pred_name = cs.class2Name(y_pred)[0]
    # generate random file name and store
    cv2.imwrite(path + pred_name + "foundimg" +
                str(random.randint(1, 99999999)) +
                ".jpg", im) # writes image test.bmp to disk

    return pred_name, dist_hyp

def drawKeypoints(im):
    surf = cv2.SURF(400)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(im, None)
    surf.hessianThreshold = 400 
    kp, des = surf.detectAndCompute(im, None)
    img_kp = cv2.drawKeypoints(im, kp, None, (0, 0, 255), 4)

    return img_kp

while(1):

    _, im = cap.read()

    im_orig = im.copy()
    im = it.resize(im, 600)

    img_height, img_width, _ = im.shape

    # draw box where we want to detect the oject
    im_detect, detect_x = createDetectBox(im, 2)
    # remove the background, find only moving part
    thresh = removeBackGround(im)
    # get contours
    pts, cnts, radius = getContours(thresh, objFound is False)

    # TODO fix this later
    # im = cv2.drawContour(im, cnts, -1, (0, 255, 0), 3)

    im_detect, movement = montionDetect(pts, im_detect)

    # TODO this block of code is SHIT. Re-write it!
    # if objFound is False and abs(movement[0]) > 20 and movement[2] > detect_x and radius > RADIUS:
        # objFound = True
        #e y_pred, dist_hyp = cs.predictImg(im)
        # # path = cs.TRAIN_PATH + cs.class2Name(y_pred)[0] + '/'
        # path = cs.TRAIN_PATH + 'tape/'
        # cv2.imwrite(path + "foundimg" + str(random.randint(1, 99999999)) + ".jpg", im) # writes image test.bmp to disk
        # y_pred = cs.class2Name(y_pred)[0]
        # a = im.copy()
    # elif abs(movement[0]) > 20 and movement[2] < detect_x and radius > RADIUS:
        # objFound = False
    # elif objFound is True:
        # cv2.putText(im_detect, "Recorded new object!",
            # (230, im_detect.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            # 0.35, (0, 0, 255), 1)

    cv2.putText(a, str(y_pred), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
    cv2.imshow('frame', drawKeypoints(im))
    cv2.imshow('frame1', thresh)
    cv2.imshow('frame2', im_detect)
    cv2.imshow('frame3', a)

    counter += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord('p'):
        pred, hyp = predictImageAndStore(im_orig)
        y_pred = pred + str(hyp)
    elif k == ord('d'):
        pred, hyp = predictImageAndStore(im, isdump=True)
        y_pred = pred + str(hyp)
        a = im.copy()
    elif k == ord('t'):

        cv2.imwrite("foundimg" +
                    str(random.randint(1, 99999999)) +
                    ".jpg", a) # writes image test.bmp to disk


cv2.destroyAllWindows()
cap.release()
