from pathlib import Path
import numpy as np
import cv2 as cv
from pydicom import dcmread
import napari

def show_image(*images):
    viewer = napari.Viewer()
    for index, image in enumerate(images):
        im, sc = image[0], image[1]
        viewer.add_image(im, scale=(sc[0], sc[1], sc[1]), name = "image"+str(index))

def window_image(image, center, width):
    left, right = center - width, center + width
    copy = image.copy()
    copy[copy < left] = left
    copy[copy > right] = right
    return copy

def edge_image(image):
    borders = np.empty_like(image)
    max_color = image.max()
    min_color = image.min()
    for i, slice_ in enumerate(image):
        slice_ -= min_color
        slice_ = (slice_ / max_color * 256).astype('uint8')
        cvt = cv.cvtColor(slice_, cv.COLOR_GRAY2RGB)
        border = cv.Canny(cvt, 100, 200)
        borders[i] = border
    return borders

def read_serie(path):
    directory = Path(path)
    slices = [dcmread(str(filename)) for filename in directory.iterdir()]
    data = [(slice_.SliceLocation, slice_.pixel_array + int(slice_.RescaleIntercept)) for slice_ in slices]
    data = sorted(data)
    image = np.array([d[1] for d in data])
    scale = slices[0].SliceThickness, slices[0].PixelSpacing[0]
    return image, scale
