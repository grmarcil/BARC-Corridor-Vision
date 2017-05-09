import cv2
import numpy as np
import skimage
import skimage.io

# Experimental description
#
# Four pieces of paper were laid out in the hallway with marked corners at
# measured distances from the BARC robot's camera. With the camera center at
# (0,0), two pieces of paper wer placed with corners marking 12" in front of the
# camera, 12" to either side ((12", 12") and (-12", 12")). The next two were
# place 60" from the camera, again 12" to either side of the camera axis ((12",
# 60"), (-12", 60")). See photo experimental_setup.jpg for clarification.
#
# A photo of this layout was captured from the robot camera and the pixel
# coordinates of each point marked with paper were measured using
# skimage.io.imshow. These points are recorded in the array img_points below.
#
# To compute a perspective transform between these sets of points, we had to
# decide on a pixel scale for the world view, and an output size for the warped
# image. The constants below were picked mostly by hand tuning so that the
# warped image still has the vehicle camera centered at the bottom of the image
# and shows as far down the hallway as possible.


# Array of points measured from image four_points.jpg
img = skimage.io.imread('four_points.jpg')
img_points = np.array([[55.2, 174.7], [258.0, 171.8], [136.6, 125.8], [179.8, 125.9]], dtype="float32")

# Constants for warped image output
w = 480
h = 640
scale = 4 #pixels per inch in top-down view
# Array of corresponding world coordinates from top down view (x,y) plane
world_points = np.array([[w/2 - 12*scale, h - 12*scale], [w/2 + 12*scale, h - 12*scale], [w/2 - 12*scale, h - 60*scale], [w/2 + 12*scale, h - 60*scale]], dtype="float32")

# Get perspective transformation matrix
# This matrix will transform from car camera view to an isometric top-down view
# of the floor (essentially a map view).
M = cv2.getPerspectiveTransform(img_points, world_points)
np.save("perspTransformMatrix.npy", M)

# Warp source image to top-down view
img_warp = cv2.warpPerspective(img, M, (w,h))
skimage.io.imsave('four_points_warp.jpg', img_warp)
