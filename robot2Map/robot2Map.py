import sys
import cv2
import numpy as np
import skimage
import skimage.io

def robot2Map(img):
    w = 480
    h = 640
    M = np.load("perspTransformMatrix.npy")
    img_warp = cv2.warpPerspective(img, M, (w,h))
    return img_warp

def main(img_file):
    img = skimage.io.imread(img_file)
    warp = robot2Map(img)
    skimage.io.imshow(warp)
    skimage.io.show()

if __name__ == '__main__':
    main(sys.argv[1])
