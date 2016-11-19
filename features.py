import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, color, exposure, io
import numpy as np
from math import atan2, sin, cos, degrees

def cv_hog():
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (16,16)
    nbins = 6
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    im = get_kitty_face_hog('CAT_DATASET/00000007_007.jpg', 'CAT_DATASET/00000007_007.jpg.cat')#cv2.cvtColor(cv2.imread('CAT_DATASET/00000001_000.jpg'), cv2.COLOR_BGR2GRAY)
    
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    h = hog.compute(im,winStride,padding,locations)
    print(h.shape, h)
    
def ski_hog():
    image = get_kitty_face_hog('CAT_DATASET/00000050_009.jpg', 'CAT_DATASET/00000050_009.jpg.cat')#color.rgb2gray(io.imread('CAT_DATASET/00000001_000.jpg'))
    

    fd, hog_image = hog(image, orientations=6, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
    print(fd.shape, fd)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')
    
    print(hog_image.shape, hog_image)
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

def draw_pts(imgFile, catFile):
    pts = np.loadtxt(catFile, dtype='int_', delimiter=' ', usecols=range(1,19))
    image = color.rgb2gray(io.imread(imgFile))
    nose = (pts[4], pts[5])
    eye_right = (pts[2], pts[3])
    eye_left = (pts[0], pts[1])
    ear_left_left = (pts[6], pts[7])
    ear_left_tip = (pts[8], pts[9])
    ear_left_right = (pts[10], pts[11])
    ear_right_left = (pts[12], pts[13])
    ear_right_tip = (pts[14], pts[15])
    ear_right_right = (pts[16], pts[17])
    
    cv2.circle(image, eye_left, 10, (0,255,0), 3)
    cv2.circle(image, eye_right, 10, (0,255,0), 3)
    cv2.circle(image, nose, 10, (0,255,0), 3)
    cv2.circle(image, ear_left_left, 10, (0,255,0), 3)
    cv2.circle(image, ear_left_tip, 10, (0,255,0), 3)
    cv2.circle(image, ear_left_right, 10, (0,255,0), 3)
    cv2.circle(image, ear_right_left, 10, (0,255,0), 3)
    cv2.circle(image, ear_right_tip, 10, (0,255,0), 3)
    cv2.circle(image, ear_right_right, 10, (0,255,0), 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    ax1.set_adjustable('box-forced')
    plt.show()
    print(pts)
    
def get_kitty_face_hog(imgFile, catFile):
    pts = np.loadtxt(catFile, dtype='int_', delimiter=' ', usecols=range(1,19))
    image = color.rgb2gray(io.imread(imgFile))
    
    eye_right = (pts[2], pts[3])
    eye_left = (pts[0], pts[1])
    nose = (pts[4], pts[5])
    ear_left_left = (pts[6], pts[7])
    ear_left_tip = (pts[8], pts[9])
    ear_left_right = (pts[10], pts[11])
    ear_right_left = (pts[12], pts[13])
    ear_right_tip = (pts[14], pts[15])
    ear_right_right = (pts[16], pts[17])
    
    #get the angle by which t0 rotate_pt
    angle = atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
    (h, w) = image.shape[:2]
    eyes_center = (w / 2, h / 2)
    
    eye_right = rotate_pt(eye_right[0], eye_right[1] ,eyes_center[0], eyes_center[1], angle)
    eye_left = rotate_pt(eye_left[0], eye_left[1] ,eyes_center[0], eyes_center[1], angle)
    nose = rotate_pt(nose[0], nose[1] ,eyes_center[0], eyes_center[1], angle)
    
    #cv2.circle(image, ear_left_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_right, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_right, 10, (0,255,0), 3)
    
    # rotate_pt the image by angle degrees
    angle = degrees(angle)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    
    #scale the image to 20 pixels between eyes
    eye_w = eye_right[0] - eye_left[0]
    
    #update scaled point for eyes
    
    new_w = (80*w)/eye_w
    
    scale = new_w/w
    
    scaled = cv2.resize(rotated,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    
    eye_left = scale_pt(eye_left[0], eye_left[1], scale)
    eye_right = scale_pt(eye_right[0], eye_right[1], scale)
    nose = scale_pt(nose[0], nose[1], scale)
    
    #cv2.circle(scaled, eye_left, 5, (0,255,0), 2)
    #cv2.circle(scaled, eye_right, 5, (0,255,0), 2)
    #cv2.circle(scaled, nose, 5, (0,255,0), 2)
    
    eyes_center = ((eye_right[0] + eye_left[0])/2, eye_left[1]+24)
    
    
    delta = 64
    roi = scaled[eyes_center[1]-delta: eyes_center[1]+delta, eyes_center[0]-delta: eyes_center[0]+delta]
    
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    return roi

def rotate_pt(x, y, x_t, y_t, angle):
    #translate to center
    x = x - x_t
    y = y - y_t
    
    #rotate_pt
    xr = y*sin(angle) + x*cos(angle)
    yr = y*cos(angle) - x*sin(angle)
    
    #translate back
    x = xr + x_t
    y = yr + y_t
    
    return (int(round(x)), int(round(y)))

def scale_pt(x, y, scale):
    return (int(round(x*scale)), int(round(y*scale)))

    
    
ski_hog()

