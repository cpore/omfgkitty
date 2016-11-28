import os, time, glob, cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure, io
import numpy as np
from math import atan2, sin, cos, degrees
from sklearn import preprocessing
from operator import itemgetter


def detect_cats(image, modelFile):
    model = np.loadtxt(modelFile, delimiter=',')
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #Detection window size. Must be aligned to block size and block stride.
    #Must match the size of the training image. Use (64, 128) for default.
    winSize = (48, 48)
    #Block size in pixels. Align to cell size. Use (16, 16) for default.
    blockSize = (8,8)
    #Block stride. Must be a multiple of cell size. Use (8,8) for default.
    blockStride = (4,4)
    #Cell size. Use (8, 8) for default.
    cellSize = (4,4)
    #Number of bins.
    nbins = 6
    #NOT USED
    derivAperture = 0
    #Gaussian smoothing window parameter.
    #winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
    winSigma = -1
    #NOT USED
    histogramNormType = 0
    #L2-Hys normalization method shrinkage.
    L2HysThreshold = 0.3
    #Do gamma correction preprocessing or not. Use true for default.
    gammaCorrection = 1
    #Maximum number of detection window increases.
    nlevels = 1024
    
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    hog.setSVMDetector(model)
    
    #Threshold for the distance between features and SVM classifying plane.
    #See gpu::HOGDescriptor::detect() for details.
    #Threshold for the distance between features and SVM classifying plane.
    #Usually it is 0 and should be specfied in the detector coefficients (as the last free coefficient).
    #But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    #hitThreshold = 1.5
    #Window stride. It must be a multiple of block stride.
    winStride = (6,6)
    #Mock parameter to keep the CPU interface compatibility. It must be (0,0).
    padding = (2,2)
    #Coefficient of the detection window increase.
    scale = 1.5
    #After detection some objects could be covered by many rectangles. 
    #This coefficient regulates similarity threshold. 0 means don't perform grouping.
    #Should be an integer if not using meanshift grouping. Use 2.0 for default
    finalThreshold = 1.5
    #
    useMeanshiftGrouping = False
    
    largest = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    
    imgscale = 336/largest #if image.shape[0] > 320 else 1 
    
    image = cv2.resize(image,None,fx=imgscale, fy=imgscale, interpolation = cv2.INTER_AREA)
    
    print(image.shape)
    found, w = hog.detectMultiScale(image, winStride=winStride, padding=padding, scale=scale, finalThreshold=finalThreshold, useMeanshiftGrouping=useMeanshiftGrouping)
    print('found: ', found, 'w', w)
    boxes = convert_to_coords(found)
    if len(w) == 0:
        cv2.imshow('rect_image',image)
        cv2.waitKey(0)
        return
    i = np.argmax(w)
    if w[i][0] < .7:
        cv2.imshow('rect_image',image)
        cv2.waitKey(0)
        return
    likely = boxes[i]
    
    
    #boxes = non_max_suppression_fast(boxes, 0.6)
    boxes = pick(boxes, likely, 0.6)
    #print(h.shape, h.ravel())
    
    print('boxes: ', boxes, 'w', w[i])
    draw_found(image, boxes)

def draw_found_max(image, found, weights):
    if len(weights) == 0:
        cv2.imshow('rect_image',image)
        cv2.waitKey(0)
        return
    i = np.argmax(weights)
    if weights[i][0] < 1:
        cv2.imshow('rect_image',image)
        cv2.waitKey(0)
        return
    pad = 0
    pts = found[i]
    cv2.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), (255,0,0), 2)
    cv2.imshow('rect_image',image)
    cv2.waitKey(0)

def draw_found(image, found):
    for i in range(found.shape[0]):
        pts = found[i]
        cv2.rectangle(image, (pts[0], pts[1]), (pts[2], pts[3]), (255,0,0), 2)
        cv2.imshow('rect_image',image)
        cv2.waitKey(100)
    cv2.imshow('rect_image',image)
    cv2.waitKey(0)
    
def convert_to_coords(boxes):
    for box in boxes:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
    return boxes

def pick(boxes, likely, overlapThresh):  
    results = []       
    for box in boxes:
        #if the boxes overlap
        overlap = area1(likely, box)
        overlapRatio = overlap/area(likely)
        if overlap > 0 and overlapRatio > overlapThresh:
            results.append((overlapRatio, box))
    
    results = max(results,key=itemgetter(0))[1]
    a = np.array([results])
        
    return a

def area1(a, b):  # returns None if rectangles don't intersect
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def area(a):
    return (a[2] - a[0]) * (a[3]-a[1])
        
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def cv_hog(image, catFile, imgFunc):
    
    #Detection window size. Must be aligned to block size and block stride.
    #Must match the size of the training image. Use (64, 128) for default.
    winSize = (48, 48)
    #Block size in pixels. Align to cell size. Use (16, 16) for default.
    blockSize = (8,8)
    #Block stride. Must be a multiple of cell size. Use (8,8) for default.
    blockStride = (4,4)
    #Cell size. Use (8, 8) for default.
    cellSize = (4,4)
    #Number of bins.
    nbins = 6
    #NOT USED
    derivAperture = 0
    #Gaussian smoothing window parameter.
    #winSigma >= 0 ? winSigma : (blockSize.width + blockSize.height)/8.;
    winSigma = 0
    #NOT USED
    histogramNormType = 0
    #L2-Hys normalization method shrinkage.
    L2HysThreshold = 0.2
    #Do gamma correction preprocessing or not. Use true for default.
    gammaCorrection = 0
    #Maximum number of detection window increases.
    nlevels = 128
    
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    im = imgFunc(image, catFile)#cv2.cvtColor(cv2.imread('CAT_DATASET/00000001_000.jpg'), cv2.COLOR_BGR2GRAY)
    
    #winStride = (8,8)
    #padding = (8,8)
    #locations = ((10,20),)
    h = hog.compute(im)
    #print(h.shape, h.ravel())
    return h.ravel()
    
def ski_hog(image, catFile, imgFunc):
    #color.rgb2gray(io.imread(imgFile))
    image = imgFunc(image, catFile)#color.rgb2gray(io.imread('CAT_DATASET/00000001_000.jpg'))
    
    orientations=6
    pixels_per_cell=(4,4)
    cells_per_block=(2,2)
    visualise=True
    
    fd, hog_image = hog(image, orientations, pixels_per_cell, cells_per_block, visualise)
    
    #print(fd.shape, fd)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray, interpolation='none')
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')
    
    #print(hog_image.shape, hog_image)
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray, interpolation='none')
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()

def draw_pts(image, catFile):
    pts = np.loadtxt(catFile, dtype='int_', delimiter=' ', usecols=range(1,19))
    
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
    
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print(pts)
    
def cv_shape():
    img = cv2.imread('CAT_DATASET/00000001_000.jpg',0)
    edges = cv2.Canny(img,100,200)
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()
    
def get_kitty_face_texture(image, catFile):
    pts = np.loadtxt(catFile, dtype='int_', delimiter=' ', usecols=range(1,19))
    
    eye_left = (pts[0], pts[1])
    eye_right = (pts[2], pts[3])
    
    nose = (pts[4], pts[5])
    
    #translate about the nose
    (h, w) = image.shape[:2]
    img_center = (int(round(w / 2)), int(round(h / 2)))
    
    nose_x = nose[0]
    nose_y = nose[1]
    
    tx = img_center[0]-nose_x
    ty = img_center[1]-nose_y
    
    M = np.float32([[1,0,tx],[0,1,ty]])
    translated = cv2.warpAffine(image,M,(w,h))
    
    eye_left = (eye_left[0]+tx, eye_left[1]+ty)
    eye_right = (eye_right[0]+tx, eye_right[1]+ty)
    
    nose = (nose[0]+tx, nose[1]+ty)
    '''
    cv2.circle(translated, eye_left, 5, (0,255,0), 2)
    cv2.circle(translated, eye_right, 5, (0,255,0), 2)
    cv2.circle(translated, nose, 5, (0,255,0), 2)
    cv2.imshow('tanslated', translated)
    cv2.waitKey(0)
    '''
    #get the angle (in radians) by which to rotate_pt
    angle = atan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0])
    
    eye_right = rotate_pt(eye_right[0], eye_right[1] ,img_center[0], img_center[1], angle)
    eye_left = rotate_pt(eye_left[0], eye_left[1] ,img_center[0], img_center[1], angle)
    nose = rotate_pt(nose[0], nose[1] ,img_center[0], img_center[1], angle)
    
    #cv2.circle(image, ear_left_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_right, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_right, 10, (0,255,0), 3)
    
    # rotate_pt the image by angle degrees
    angle = degrees(angle)
    M = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated = cv2.warpAffine(translated, M, (w, h))
    
    
    #scale the image to 20 pixels between eyes
    eye_w = eye_right[0] - eye_left[0]
    
    #update scaled point for eyes
    
    new_w = (30*w)/eye_w
    
    scale = new_w/w
    
    scaled = cv2.resize(rotated,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    
    eye_left = scale_pt(eye_left[0], eye_left[1], scale)
    eye_right = scale_pt(eye_right[0], eye_right[1], scale)
    nose = scale_pt(nose[0], nose[1], scale)
    
    #add border to ensure faces on edges of photos don't cause error due to roi being out of image boundaries
    border = 48
    scaled = cv2.copyMakeBorder(scaled,border,border,border,border,cv2.BORDER_CONSTANT,value=(0,0,0))
    eye_left = (eye_left[0]+border, eye_left[1]+border)
    eye_right = (eye_right[0]+border, eye_right[1]+border)
    nose = (nose[0]+border, nose[1]+border)
    '''
    cv2.circle(scaled, eye_left, 5, (0,255,0), 2)
    cv2.circle(scaled, eye_right, 5, (0,255,0), 2)
    cv2.circle(scaled, nose, 5, (0,255,0), 2)
    '''
    eyes_center = ((int(round(eye_right[0] + eye_left[0])/2)), eye_left[1]+9)
    
    delta = int(border/2)
    roi = scaled[eyes_center[1]-delta: eyes_center[1]+delta, eyes_center[0]-delta: eyes_center[0]+delta]
    
    #print(roi.shape)
    #cv2.imshow("ROI", roi)
    #cv2.waitKey(0)
    return roi

def get_kitty_face_shape(image, catFile):
    pts = np.loadtxt(catFile, dtype='int_', delimiter=' ', usecols=range(1,19))
    
    nose = (pts[4], pts[5])
    ear_left_tip = (pts[8], pts[9])
    ear_right_tip = (pts[14], pts[15])
    
    #translate about the nose
    (h, w) = image.shape[:2]
    img_center = (int(round(w / 2)), int(round(h / 2)))
    
    nose_x = nose[0]
    nose_y = nose[1]
    
    tx = img_center[0]-nose_x
    ty = img_center[1]-nose_y
    
    M = np.float32([[1,0,tx],[0,1,ty]])
    translated = cv2.warpAffine(image,M,(w,h))
    
    ear_left_tip = (ear_left_tip[0]+tx, ear_left_tip[1]+ty)
    ear_right_tip = (ear_right_tip[0]+tx, ear_right_tip[1]+ty)
    
    nose = (nose[0]+tx, nose[1]+ty)
    '''
    cv2.circle(translated, ear_left_tip, 5, (0,255,0), 2)
    cv2.circle(translated, ear_right_tip, 5, (0,255,0), 2)
    cv2.circle(translated, nose, 5, (0,255,0), 2)
    cv2.imshow('tanslated', translated)
    cv2.waitKey(0)
    '''
    #get the angle (in radians) by which to rotate_pt
    angle = atan2(ear_right_tip[1] - ear_left_tip[1], ear_right_tip[0] - ear_left_tip[0])
    
    ear_right_tip = rotate_pt(ear_right_tip[0], ear_right_tip[1] ,img_center[0], img_center[1], angle)
    ear_left_tip = rotate_pt(ear_left_tip[0], ear_left_tip[1] ,img_center[0], img_center[1], angle)
    nose = rotate_pt(nose[0], nose[1] ,img_center[0], img_center[1], angle)
    
    #cv2.circle(image, ear_left_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_left_right, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_left, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_tip, 10, (0,255,0), 3)
    #cv2.circle(image, ear_right_right, 10, (0,255,0), 3)
    
    # rotate_pt the image by angle degrees
    angle = degrees(angle)
    M = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    rotated = cv2.warpAffine(translated, M, (w, h))
    
    
    #scale the image to 36 pixels between ear tips
    ear_w = ear_right_tip[0] - ear_left_tip[0]
    
    #update scaled point for eyes
    
    new_w = (38*w)/ear_w
    
    scale = new_w/w
    
    scaled = cv2.resize(rotated,None,fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
    
    
    ear_left_tip = scale_pt(ear_left_tip[0], ear_left_tip[1], scale)
    ear_right_tip = scale_pt(ear_right_tip[0], ear_right_tip[1], scale)
    nose = scale_pt(nose[0], nose[1], scale)
    
    #add border to ensure faces on edges of photos don't cause error due to roi being out of image boundaries
    border = 48
    scaled = cv2.copyMakeBorder(scaled,border,border,border,border,cv2.BORDER_CONSTANT,value=(0,0,0))
    ear_left_tip = (ear_left_tip[0]+border, ear_left_tip[1]+border)
    ear_right_tip = (ear_right_tip[0]+border, ear_right_tip[1]+border)
    nose = (nose[0]+border, nose[1]+border)
    
    #cv2.imshow("scaled", scaled)
    #cv2.waitKey(0)
    '''
    cv2.circle(scaled, ear_left_tip, 5, (0,255,0), 2)
    cv2.circle(scaled, ear_right_tip, 5, (0,255,0), 2)
    cv2.circle(scaled, nose, 5, (0,255,0), 2)
    '''
    ears_center = ((int(round(ear_right_tip[0] + ear_left_tip[0])/2)), ear_left_tip[1]+22)
    
    delta = int(border/2)
    roi = scaled[ears_center[1]-delta: ears_center[1]+delta, ears_center[0]-delta: ears_center[0]+delta]
    
    #print(roi.shape)
    #cv2.imshow("ROI", roi)
    #cv2.waitKey(0)
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

def show_faces():
    for filename in glob.glob('CAT_DATASET/*.jpg'):
        catFile = filename +'.cat'
        print(filename)
        get_kitty_face_texture(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY), catFile)
        
def get_kitty_negative(image, catFile):
    (h, w) = image.shape[:2]
    img_center = (int(round(w / 2)), int(round(h / 2)))
    
    #add border to ensure faces on edges of photos don't cause error due to roi being out of image boundaries
    border = 48
    newimg = cv2.copyMakeBorder(image,border,border,border,border,cv2.BORDER_CONSTANT,value=(0,0,0))
    
    delta = int(border/2)
    roi = newimg[img_center[1]-delta: img_center[1]+delta, img_center[0]-delta: img_center[0]+delta]
    
    #print(roi.shape)
    #cv2.imshow("ROI", roi)
    #cv2.waitKey(0)
    return roi
        
def make_features():
    f = open('data/hog_pos_tex.data','w')
    numFiles = len([name for name in os.listdir('CAT_DATASET/')])/2
    done = 0
    for filename in glob.glob('CAT_DATASET/*.jpg'):
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        catFile = filename +'.cat'
        desc = cv_hog(image, catFile, get_kitty_face_texture)
        desc = np.append(np.array([1]), desc)
        descString = ','.join(['%.8f' % num for num in desc])
        print(descString, file=f)
        if get_time() % 100 == 0:
            print('processed...' + str(done) + ' of ' + str(numFiles))
        done += 1
        
def make_negative_features():
    f = open('data/hog_neg_tex.data','w')
    numFiles = len([name for name in os.listdir('VOC_NEGATIVES/')])
    done = 0
    for filename in glob.glob('VOC_NEGATIVES/*.jpg'):
        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        desc = cv_hog(image, None, get_kitty_negative)
        desc = np.append(np.array([0]), desc)
        descString = ','.join(['%.8f' % num for num in desc])
        print(descString, file=f)
        if get_time() % 100 == 0:
            print('processed...' + str(done) + ' of ' + str(numFiles))
        done += 1
        
def get_time():
    return int(round(time.time() * 1000))

def show_detected():
    for filename in glob.glob('VOC_NEGATIVES/*.jpg'):
        detect_cats(cv2.imread(filename), 'models/svm_tex_spark1.model')

show_detected()
#cv_hog()
#ski_hog()
# 
# imgFile3 = 'CAT_DATASET/00000298_014.jpg'
# catFile3 = imgFile3 +'.cat'
#detect_cats(cv2.imread('CAT_DATASET/00000009_016.jpg'), 'models/svm2.model')
# imgFile2 = 'CAT_DATASET/00000156_002.jpg'
# catFile2 = imgFile2 +'.cat'
#imgFile = 'CAT_DATASET/00000032_002.jpg'
#catFile = imgFile +'.cat'
# image = color.rgb2gray(io.imread(imgFile3))
#cv_hog(cv2.cvtColor(cv2.imread(imgFile), cv2.COLOR_BGR2GRAY), catFile, get_kitty_face_shape)

#make_negative_features()
#make_features()
#get_kitty_face_texture(image, catFile3)
#get_kitty_face_shape(image, catFile3)
#make_features()

# if __name__ == '__main__':
#     sc = pyspark.SparkContext()
#     #sc.setLogLevel('WARN')
#       
#     log4jLogger = sc._jvm.org.apache.log4j
#     LOGGER = log4jLogger.LogManager.getLogger(__name__)

