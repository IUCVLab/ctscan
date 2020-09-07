import skimage, os
from skimage.transform import resize
from pydicom import dcmread
from pydicom.filereader import read_dicomdir
from os.path import dirname, join
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cv2 import erode, dilate
import math
from sklearn.cluster import DBSCAN as Dbscan
import glob
import sys
import math
import pandas as pd
from skimage.io import imread
from skimage import color, io
get_ipython().run_line_magic('matplotlib', 'inline')


def get_data(inpath, csv_path):
    
    dicom_dir = read_dicomdir(filepath)
    base_dir = dirname(filepath)
    image_files = [join(base_dir, *image_desc.ReferencedFileID) for image_desc in dicom_dir.patient_records[0].children[0].children[3].children]
    slices = [dcmread(filename) for filename in image_files]
    images = np.array([sl.pixel_array for sl in slices])

    labels = pd.read_csv(csv_path)
    labels = labels.label.tolist()
    
    return images, labels


def segment_contours(imgg):

    original = imgg
    imgg = cv2.medianBlur(imgg, 5)
    feature_array = np.empty((512,512,3))

    for x in range(512):
        for y in range(512):
            feature_array[x][y] = (x, y, (imgg[x][y] - 1000) * 5 if 1000 < imgg[x][y] < 1100 else 0)

    db = Dbscan(eps = 11.4, min_samples=15)
    predicted = db.fit_predict(feature_array.flatten().reshape(-1, 3)).reshape(512,512)

    img = predicted.astype('uint8')
    kernel = np.ones((7, 7),dtype=np.uint8)
    img=erode(img,kernel)
    img = cv2.medianBlur(img, 7)

    contour_image = np.zeros(shape=[512, 512], dtype=np.uint8)
    contour_dilated = np.zeros(shape=[512, 512], dtype=np.uint8)
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,1)
    cir = 0

    for c in contours:

        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)

        M = cv2.moments(c)  

        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        except ZeroDivisionError:
            cX = int(M["m10"])
            cY = int(M["m01"])

        if perimeter == 0 or (area > 3500 or area < 200) or (cX < 220 or cX > 300) or (cY < 180 or cY > 300):
            continue

        else:

            (x,y),radius = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            circularity = 4*math.pi*(area/(perimeter*perimeter))

            if 0.45 < circularity < 1.2:
                cir +=1
                cv2.drawContours(contour_image, [c], -1, (255, 255, 255), 3)

    if cir > 0:
        
        new_contour = contour_image
        new_contour = dilate(new_contour, kernel)
        contours_new, hierarchy = cv2.findContours(new_contour, cv2.RETR_TREE, 1)
        cv2.drawContours(contour_dilated, contours_new[0], -1, (255, 255, 0), 3)
        
        return 1, contour_dilated

    else:
        return 0, contour_dilated
    
    
def get_contours(images, labels):
    
    contour_lines = []
    for i in range(len(images)):

        imgg = images[i]
        label = labels[i]
        test, contour_image = segment_contours(imgg)

        if (test == label) and (label == 1):
            
            cs, _ = cv2.findContours(contour_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cmin = min(cs, key=cv2.contourArea)
            contour_lines.append(cmin)
            
    return contour_lines
            
    
def triangulation(contours):
    
    def get_center(contour):
        x, y = contour[0, 0, 0], (max(contour[:, 0, 1]) + min(contour[:, 0, 1])) / 2
        return (x, y)

    def angle(center, x, y):
        if center[0] == x:
            return np.pi // 2 if center[1] < y else -np.pi // 2
        dy = y - center[1]
        dx = x - center[0]
        tg = dy / dx
        return math.atan(tg) + (np.pi if dx < 0 else 0)

    cl = contour_lines
    vertices = []
    faces = []
    vertices_offset = 0

    for level, (a, b) in enumerate(zip(cl[:-1], cl[1:])):
        ca, cb = get_center(a), get_center(b)

        for i in range(a.shape[0]):
            vertices.append([a[i, 0, 0], a[i, 0, 1], level])
        for j in range(b.shape[0]):
            vertices.append([b[j, 0, 0], b[j, 0, 1], level+1])

        i, j = 0, 0  
        
        while i <= a.shape[0] and j <= b.shape[0]:
            
            aa = angle(ca, a[i % a.shape[0], 0, 0], a[i % a.shape[0], 0, 1])
            ab = angle(cb, b[j % b.shape[0], 0, 0], b[j % b.shape[0], 0, 1])

            if i == a.shape[0]:
                while j < b.shape[0]:
                    faces.append([3, vertices_offset, vertices_offset + a.shape[0] + j, vertices_offset + a.shape[0] + (j + 1) % b.shape[0]])
                    j += 1
                break

            if j == b.shape[0]:
                while i < a.shape[0]:
                    faces.append([3, vertices_offset + a.shape[0], vertices_offset + i, vertices_offset + (i + 1) % a.shape[0]])
                    i += 1
                break

            if aa < ab:
                faces.append([3, vertices_offset + i, vertices_offset + (i + 1) % a.shape[0], vertices_offset + a.shape[0] + j])
                i += 1
            else:
                faces.append([3, vertices_offset + i, vertices_offset + a.shape[0] + j, vertices_offset + a.shape[0] + (j + 1) % b.shape[0]])
                j += 1

        vertices_offset += a.shape[0] + b.shape[0]

        return np.array(vertices), np.hstack(faces)
    
    
def visulization(vertices, faces, params=None):
    
    
    # TO be tweaked considering Tkinter Visulization
    
    
def pipeline(inpath, csv_path, params=None):
    
    images, labels = get_data(inpath, csv_path)
    contours = get_contours(images, labels)
    vertices, faces = triangulation(contours)
    visulazation(vertices, faces, params)
    

if __name__ == "__main__":
    
    
    images_path = 'D:\\DS PROJECT\\CT\\01_Samodurov\\DICOMDIR'
    labels_csv = 'D:\\Aorta_dataset\\aorta_dataset.csv'
    params = []
    pipeline(images_path, labels_csv, params)
    