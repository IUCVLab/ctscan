from pathlib import Path
import numpy as np
from pydicom import dcmread
import napari
from src.process.range import window_image

import numpy as np
import ipywidgets as ipyw

try:
    import matplotlib.pyplot as plt
except: pass
try:
    import plotly.graph_objects as go
except: pass


class VolumeSliceViewer:
    
    def __init__(self, volume, window=(None,None), renderlib = 'matplotlib'):
        self.volume = volume
        self.zmin, self.zmax = window[0], window[1]
        self.renderlib = renderlib
        
        if renderlib not in ['matplotlib', 'plotly']:
            raise ValueError("renderlib must be one of two options: matplotlib | plotly")
        
        ipyw.interact(self.view_selection, view=ipyw.RadioButtons(
            options=['x-y','x-z', 'y-z'], value='x-y', 
            description='Slice plane selection:', disabled=False,
            style={'description_width': 'initial'}))
    
    def view_selection(self, view):
        orient = {"x-y": ([1,2,0], False), "x-z": ([0,2,1], [0]), "y-z": ([0,1,2], [0,1])}
        opts = orient[view]
        self.vol = np.transpose(self.volume, opts[0])
        if opts[1]:
            self.vol = np.flip(self.vol, axis=opts[1])
        maxZ = self.vol.shape[2] - 1
        
        ipyw.interact(self.plot_slice, 
            z=ipyw.IntSlider(min=0, max=maxZ, step=1, continuous_update=False, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        if self.renderlib == "matplotlib":
            plt.figure(figsize=(10,10))
            plt.imshow(self.vol[:,:,z], cmap='gray', 
                vmin=self.zmin, vmax=self.zmax)
        elif self.renderlib == "plotly":
#             fig = px.imshow(self.vol[:,:,z], color_continuous_scale='gray', width=700, height=700, 
#                 zmin=self.zmin, zmax=self.zmax)
#             fig.show()
            fig = go.Figure(data=go.Image(self.vol[:,:,z],
                                          zmin=self.zmin, zmax=self.zmax, color_continuous_scale='gray'),
                            width=700, height=700)
            fig.show()


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
