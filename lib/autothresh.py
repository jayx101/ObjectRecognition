import cv2
import imagetransform as it
import os
import numpy as np

def cropCoords(cnts):
    '''
    Get the contours and find the crop areas
    '''
    sort_cnts = [(cv2.contourArea(cnts[i]), i) for i in range(len(cnts))]
    sort_cnts = sorted(sort_cnts, key=lambda x: x[0], reverse=True)
    large_cnt_ix = sort_cnts[0][1]

    print 'cnts', sort_cnts
    print len(cnts), 'number of contours'

    # Find thre x and y cropping points
    cnts = cnts[large_cnt_ix].reshape(len(cnts[large_cnt_ix]), 2)
    cnt_tup = zip(*cnts)

    feather = 20 # pixels to add around corners
    x1, x2 = min(cnt_tup[0]) - feather, max(cnt_tup[0]) + feather
    y1, y2 = min(cnt_tup[1]) - feather, max(cnt_tup[1]) + feather

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    # dict_coord = {'x1':x1, 'x2':x2, 'y1':y1, 'y2',y2}
    return x1, x2, y1, y2

def cropImage(im, disp_steps=False):

    imgs = []

    im_height, im_width = im.shape
    # print path + imp

    l, u, v = auto_canny(im, 0.5)

    # find the threshold and caclulate the contours
    ret, thresh = cv2.threshold(im.copy(), u+20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_disp = thresh.copy()
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find the crop areas
    x1, x2, y1, y2 = cropCoords(cnts)

    # if the image is black you have to invert the crop values
    if (x1 <= 0 and y1 <= 0) and x2 > im_width and y2 > im_height:
        ret, thresh = cv2.threshold(im.copy(), u, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh_disp = thresh.copy()
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x1, x2, y1, y2 = cropCoords(cnts)

    # Create new image and draw contours
    imc = im.copy()
    cv2.drawContours(imc, cnts, -1, (0, 0, 255), 3)

    crop_img = im[y1:y2, x1:x2] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

    if disp_steps:
        imgs.append(im)
        imgs.append(thresh_disp)
        imgs.append(imc)
        imgs.append(crop_img)

    return crop_img, imgs

def auto_canny(image, sigma=0.33):
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

def testCrops(path):
    for imp in os.listdir(path):
        im = cv2.imread(path + imp, 0)
        _, imgs = cropImage(im, True)
        it.display_multi(imgs)
testCrops('../dataset/dump/')

