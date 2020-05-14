def window_image(image, center, width):
    left, right = center - width, center + width
    copy = image.copy()
    copy[copy < left] = left
    copy[copy > right] = right
    return copy

def normalize(image):
    res = image.copy()
    min_ = image.min()
    max_ = image.max()
    res = (res - min_).astype('float32') * 255 / (max_ - min_)
    return res.astype('uint8')
