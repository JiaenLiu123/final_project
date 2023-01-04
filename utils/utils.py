import cv2
import numpy as np
import io
import base64

'''
    This function takes an image and resizes it to the given size.
    If the image is smaller than the given size, it is returned as it is.

    TODO: This function needs to be improved to .

    Parameters:
        image (PIL Image): The image to be resized.
        size_x (int): The width of the resized image.
        size_y (int): The height of the resized image.
'''
def resize_image(image, size_x, size_y):
    if image.size[0] > size_x and image.size[1] > size_y:
        image = image.resize((size_x, size_y))
    elif image.size[0] > size_x and image.size[1] < size_y:
        image = image.resize(size_x)
    elif image.size[1] > size_y and image.size[0] < size_x:
        image = image.resize(size_y)
    return image

def get_image_download_link(img, filename, text):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

'''
    This function takes an image and removes the shadows from it.

    Parameters:
        image (numpy.ndarray): The image from which shadows are to be removed.
'''
def remove_shadows(image):
    # convert the image to grayscale and blur it
    rgb_planes = cv2.split(image)
    # result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        # result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    # result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm