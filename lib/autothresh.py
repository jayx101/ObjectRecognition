import cv2
import imagetransform as it
import os
import numpy as np

def _autoCanny(image, sigma=0.33):
    '''
    automatically find the threshhold values
    returns: lower, uppver and median
    '''
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return lower, upper, v

def _cropCoords(cnts, feather, min_area=50):
    '''
    Get the contours and find the crop areas.
    Looks at all found contours, combines then and returns x and y coords

    Params:
        cnts: detected contours
        feather: how much space to add around the image before crop
        min_area: specify minimum area of a contount to include
    Return:
        x1, x2, y1, y2 co ordinates
    '''

    filt_cnts = []
    for cnt in cnts:
        if cv2.contourArea(cnt) > min_area:
            cnt_re = cnt.reshape(len(cnt), 2)
            filt_cnts.append(cnt_re)

    cnts = np.concatenate(filt_cnts, axis=0)

    # find the x and y cropping points
    # cnts = cnts[large_cnt_ix].reshape(len(cnts[large_cnt_ix]), 2)
    cnt_tup = zip(*cnts)

    # pixels to add around corners
    x1, x2 = min(cnt_tup[0]) - feather, max(cnt_tup[0]) + feather
    y1, y2 = min(cnt_tup[1]) - feather, max(cnt_tup[1]) + feather

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    # dict_coord = {'x1':x1, 'x2':x2, 'y1':y1, 'y2',y2}

    return x1, x2, y1, y2

def getStaticImageCoordinates(im, upper_thresh):

    # extract image dimensions
    im_height, im_width = im.shape

    # find the threshold and caclulate the contours
    ret, thresh = cv2.threshold(im.copy(), upper_thresh, 255, cv2.THRESH_BINARY)
    thresh_disp = thresh.copy()
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the crop areas
    x1, x2, y1, y2 = _cropCoords(cnts)

    # if the image is black you have to invert the crop values
    if (x1 <= 0 and y1 <= 0) and x2 > im_width and y2 > im_height:
        ret, thresh = cv2.threshold(im.copy(), upper_thresh, 255, cv2.THRESH_BINARY_INV)
        thresh_disp = thresh.copy()
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, x2, y1, y2 = _cropCoords(cnts)

    return x1, x2, y1, y2, thresh_disp, cnts

def _createRedMask(im):
    '''
    Takes an image and removes red color
    Reurns:
        Mask showing white not red and black where red was
    '''
    # remove noise and add blur
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    im = cv2.GaussianBlur(im, (9, 9), 0)

    # create NumPy arrays from the boundaries
    lower, upper = [5, 5, 70], [150, 118, 250]
    # lower, upper = [20, 20, 70], [125, 99, 250]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(im, lower, upper)

    return mask.copy()

def _cropImage(im, x1, x2, y1, y2):

    # l, u, v = _autoCanny(im, 0.2)

    # x1, x2, y1, y2 = _cropCoords(cnts)
    # separate image from static background
    # x1, x2, y1, y2, thresh_disp, cnts = getStaticImageCoordinates(im, u)
    # Create new image and draw contours

    crop_img = im[y1:y2, x1:x2] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    return crop_img

def cropRedImage(im, disp_all=False):
    '''
    takes an image with red background and crops out the red
    Params:
        im: image to be cropped
        disp_all: will output all images involved in the crop process
    Return:
        cropped image
    '''

    mask = _createRedMask(im)
    mask_inv = 255 - mask
    imgs = []
    imgs.append(mask)
    imgs.append(mask_inv.copy())
    imgs.append(im)
    im_rect = im.copy()

    # find contours, combine them, get coordinates of objects
    _, cnts, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x1, x2, y1, y2 = _cropCoords(cnts, feather=20, min_area=100)

    # draw contours
    imc = im.copy()
    cv2.drawContours(imc, cnts, -1, (0, 0, 255), 3)
    imgs.append(imc)

    # draw rectangle around crop area
    cv2.rectangle(im_rect, (x1, y1), (x2, y2), (0, 255, 0), 3)
    imgs.append(im_rect)

    # crop image
    im_crop = _cropImage(im, x1, x2, y1, y2)
    imgs.append(im_crop)

    # display all images
    if disp_all: it.display_multi(imgs, False, cols=2)

    return im_crop

def testCrops(path):
    for imp in os.listdir(path):
        print imp
        im = cv2.imread(path + imp)
        cropRedImage(im, True)
        # _cropImage(im, True)
        # it.display_multi(imgs)

# testCrops('../dataset/train/trigger/')
