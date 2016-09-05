from lib import imagetransform
import os 
import cv2

img = cv2.imread('generateimgs/remote.jpg', 0)
# img = cv2.imread('researchimages/square.jpg', 0)
# img = cv2.imread('remote.jpg', cv2.CV_LOAD_IMAGE_UNCHANGED)
img = imagetransform.resize(img)
imgs = imagetransform.rotate_images(img, 10)
# imagetransform.display_multi(imgs, imgisgray=True)
imagetransform.write_imgs('generateimgs/', imgs, 'remote')

img = cv2.imread('generateimgs/tv.jpg', 0)
img = imagetransform.resize(img)
imgs = imagetransform.rotate_images(img, 10)
# imagetransform.display_multi(imgs, imgisgray=True)
imagetransform.write_imgs('generateimgs/', imgs, 'tv')

path = '../bag-of-words/dataset/test/tv/'
filelist = [f for f in os.listdir(path) if f.endswith('*')]
for f in filelist:
    os.remove(f)
    
    img = cv2.imread('generateimgs/remote.jpg', 0)
    
