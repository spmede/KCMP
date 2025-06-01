
import random
import requests
import numpy as np
import pycocotools.mask as maskUtils 

from io import BytesIO
from PIL import Image, ImageDraw
from skimage import measure  # Efficient edge detection


def load_image_llava(image_file):
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    if isinstance(image_file, str) and image_file.startswith(("http", "https")):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")

def load_images(image_files):
    return [load_image_llava(f) for f in image_files]

def safe_prompts(prompts):
    return [p if "<ImageHere>" in p else f"<ImageHere> {p}" for p in prompts]


def mask_object(input_img, sam_info):
    '''add balck mask'''

    # get box info
    box_info = sam_info['bbox']
    masked_image = input_img.copy()
    draw = ImageDraw.Draw(masked_image)


    if input_img.mode == "L": 
        fill_color = 0
    elif input_img.mode == "RGB":
        fill_color = (0, 0, 0)
    elif input_img.mode == "RGBA":
        fill_color = (0, 0, 0, 255)
    else:
        raise ValueError(f"Unsupported image mode: {input_img.mode}")
    
    draw.rectangle(box_info, fill=fill_color)
    
    return masked_image


def box_object(input_img, sam_info, box_color='red', box_width=3):
    '''box an object'''

    # get box info
    box_info = sam_info['bbox']
    # Create a drawable image
    image_with_box = input_img.copy()
    draw = ImageDraw.Draw(image_with_box)
    # Draw the bounding box
    draw.rectangle(box_info, outline=box_color, width=box_width)

    return image_with_box


def outline_object(input_img, sam_info, outline_color='red', line_width=3):
    '''outline an object'''

    mask = maskUtils.decode(sam_info['segmentation'])  # Binary mask (0: background, 1: object)
    contours = measure.find_contours(mask, 0.5)  # Threshold at 0.5 to get the boundary
    outlined_image = input_img.copy()
    draw = ImageDraw.Draw(outlined_image)

    for contour in contours:
        contour = [(int(x), int(y)) for y, x in contour]  
        draw.line(contour, fill=outline_color, width=line_width)

    return outlined_image


def turn_grayscale_image(input_img, manner='1'):

    # method 1
    if manner == 'default':
        gray_image = input_img.convert("L")
    
    elif manner == '3channel':
        gray_image = input_img.convert("L").convert("RGB")

    # method 2
    elif manner == 'average':
        image_array = np.array(input_img)
        gray_array = np.mean(image_array, axis=2).astype(np.uint8)
        gray_image = Image.fromarray(gray_array)

    return gray_image