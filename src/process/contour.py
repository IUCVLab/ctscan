import numpy as np
import cv2
from numba import njit
from scipy.spatial import distance

def max_width(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return 2*radius

def perimetr(contour):
    return cv2.arcLength(contour, True)

def area(contour):
    return cv2.contourArea(contour)

def center(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cY, cX

def scale(contour, space):
    scaled = contour.copy()
    scaled[...] *= space
    return scaled
