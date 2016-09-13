import cv2
import time
import matplotlib.pyplot as plt
import math
import os

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
    return rotated_mat

def rotate_images(img, degsplit=10):
    '''Takes an image and rotates it by degrees specified
    Params:
        degsplit: amt of degrees to capture each image at
    Returns:
        list of images
    '''
    imgs = []
    for i in range(360 / degsplit):
        split = i * degsplit
        rotated = rotate_image(img, split)
        imgs.append(rotated)
    return imgs

def resize(img, width=300):
    '''
    Description:
        Resizes the image based on the width. Keeps image proportions
    Params:
            img: image you would like to resize
            width: relative width you would like the image to retain
    '''

    width = float(width)
    r = width / img.shape[1]
    dim = (int(width), int(img.shape[0] * r))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def display(img, fsize=(10, 8), axis='off', imgisgray=True):
    '''
    Display a single image using matplotlib
    Params:
        fsize: matplotlib figsize
        axis: should matplotlib draw the axis or not
        convcolor: if image  is gray will handle the BGR converstion
    '''
    plt.figure(figsize=fsize)
    if imgisgray:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis(axis)
    plt.show()

def display_multi(imgs, imgisgray=True):
    """Dipslays images in a gridspec using Matplotlib
    Params:
        imgs: list of images
        convcolor: if the image is gray, it will convert grom BGR to RGB
    """
    import matplotlib.gridspec as gridspec

    cols = 4
    rows = math.ceil(((len(imgs) + 0.0) / cols))
    rows = int(rows)
    gs = gridspec.GridSpec(rows, cols, bottom=0., right=1., left=0., hspace=0., wspace=0.)
    i = 0

    for g in gs:
        ax = plt.subplot(g)
        if imgisgray:
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)
        else:
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
        ax.imshow(imgs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        if len(imgs) - 1 > i:
            i += 1
        else:
            break
    plt.show()

def write_imgs(path, imgs, prefix):
    """Write a list of images to disk
    """
    i = 0

    filelist = [f for f in os.listdir(path) if f.endswith(prefix + '*')]
    for f in filelist:
        os.remove(f)

    for img in imgs:
        cv2.imwrite(path + prefix + str(i) + '.png', img)
        print path + prefix + str(i) + '.jpg'
        i += 1

# img = cv2.imread('../remote.jpg', cv2.CV_LOAD_IMAGE_UNCHANGED)

# # img = cv2.imread('../remote.jpg')
# # img = resize(img)
# # img = rotate_image(img, 45)

# # plt.imshow(img)
# # plt.show()

# # imgs = rotateImg(img)
# imgs = rotate_images(img, 45)
# # display(img, convcolor=False)
# # write_img('imgstest/', imgs, 'rotation')
# display_multi(imgs, convcolor=False)
