import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from math import pi as PI
from skimage import measure
from skimage.segmentation import morphological_chan_vese, checkerboard_level_set
from src.process.contour import scale, center, area, max_width

def morphological_ACWE_evolution(image):

    image = cv2.medianBlur(image, 5)

    feature_array = np.empty((512,512,3))
    feature_array[..., 0] = np.array(range(image.shape[0])).reshape(-1,1)
    feature_array[..., 1] = np.array(range(image.shape[1]))
    feature_array[..., 2] = np.where(np.logical_and(0 < image, image < 100), image * 5, 0)

    db = DBSCAN(eps = 15, min_samples=15)
    image = db.fit_predict(feature_array.reshape(-1, 3)).reshape(512,512)
    
    image = image.astype('uint8')
    kernel = np.ones((2,2),dtype=np.uint8) # this must be tuned 
    image=cv2.erode(image,kernel)
    image = cv2.medianBlur(image, 5)
    
    contours,hierarchy = cv2.findContours(image,cv2.RETR_TREE,1)
    cir = 0

    for c in contours:
        
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)

        # GET center of contours
        M = cv2.moments(c)  
        
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
        except ZeroDivisionError:
            cX = int(M["m10"])
            cY = int(M["m01"])
        
        if not (perimeter == 0 or (area > 2000 or area < 500) or (cX < 220 or cX > 320) or (cY < 250 or cY > 300)):
            
            (x,y),radius = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            circularity = 4*PI*(area/(perimeter*perimeter))

            if 0.2 < circularity < 1.2:
                cir +=1
                cv2.drawContours(image, [c], -1, (255, 255, 0), 3)

    # Morphological ACWE
    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    ls = morphological_chan_vese(image, 37, init_level_set=init_ls, smoothing=3)

    return ls


def by_point(image, point):

    binary = image > 150

    labeled = measure.label(binary, connectivity=1)
    label = labeled[point[0], point[1]]
    binary = labeled == label

    return binary

def analyze_aorta(image, space, point1, point2):
    segment = np.zeros_like(image, dtype='bool')
    height = (point2[0] - point1[0]) * space[2]
    start_height = point1[0]
    volume = 0
    cent = point1[1:3]
    labels = by_point(image[start_height], cent)
    segment[start_height] = labels
    contours, _ = cv2.findContours(labels.astype('uint8'), cv2.RETR_TREE, 1)
    contour = contours[0]
    contour = contour.astype('float32')
    cent = center(contour)
    contour = scale(contour, space[:2])
    width = max_width(contour)
    prev_area = area(contour)
    max_width_slice = start_height
    max_cent = cent

    for i in range(1, point2[0] - point1[0] + 1):
        labels = by_point(image[start_height+i], cent)
        if i == 0:
            plt.imshow(labels)
        segment[start_height+i] = labels
        contours, _ = cv2.findContours(labels.astype('uint8'), cv2.RETR_TREE, 1)
        contour = contours[0].astype('float32')
        cent = center(contour)
        contour = scale(contour, space[:2])
        (x, y), w = cv2.maxEnclosingCircle(contour)
        if w > width:
            width = w
            max_cent = x,y
            max_width_slice = start_height + i
        a = area(contour)
        max_area = a if a > prev_area else prev_area
        min_area = a if a < prev_area else prev_area
        volume += ((3*max_area + min_area) / 4) * space[2]
        prev_area = a

    return width, max_width_slice, max_cent, height, volume, segment.astype('int8')
