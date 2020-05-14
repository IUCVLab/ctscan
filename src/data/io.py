from pathlib import Path
import numpy as np
import cv2 as cv
from pydicom import dcmread
import napari
from src.process.range import window_image

def show_image(*images, space=(1,1,1), window = None):
    viewer = napari.Viewer()
    for index, image in enumerate(images):
        if window is not None:
            image = window_image(image, window[0], window[1])
        viewer.add_image(image, scale=(space[2], space[0], space[1]), name = "image"+str(index))

def read_series(path, start=None, stop=None, step=None):

    directory = Path(path)
    slices = [dcmread(str(filename)) for filename in directory.iterdir()]
    data = [(slice_.SliceLocation, slice_) for slice_ in slices]
    data = sorted(data)
    
    if start is None:
        start = 0
    if stop is None:
        stop = len(slices)
    if step is None:
        step = 1
    if start < 0:
        start += len(slices)
    if stop < 0:
        stop += len(slices)
    r = range(start, stop, step)

    image = np.array([data[i][1].pixel_array + data[i][1].RescaleIntercept for i in r], dtype=np.int16)
    scale = float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[0].SliceThickness) * abs(step)
    
    return image, scale
