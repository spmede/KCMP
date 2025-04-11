
import requests
import numpy as np
import pycocotools.mask as maskUtils  # 用于解码 RLE 格式

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
    '''对图像中某物体施加黑色 mask'''

    # get box info
    box_info = sam_info['bbox']
    # 转换为可编辑模式
    masked_image = input_img.copy()
    draw = ImageDraw.Draw(masked_image)

    # 获取图片尺寸
    # img_width, img_height = input_img.size

    # 选择正确的 fill 颜色
    if input_img.mode == "L":  # 灰度图像
        fill_color = 0
    elif input_img.mode == "RGB":  # RGB 图像
        fill_color = (0, 0, 0)
    elif input_img.mode == "RGBA":  # RGBA 图像
        fill_color = (0, 0, 0, 255)
    else:
        raise ValueError(f"Unsupported image mode: {input_img.mode}")
    
    # 绘制黑色矩形 (掩盖物体)
    draw.rectangle(box_info, fill=fill_color)
    
    return masked_image


def box_object(input_img, sam_info, box_color='red', box_width=3):
    '''返回 box 框住物体的图像'''

    # get box info
    box_info = sam_info['bbox']
    # Create a drawable image
    image_with_box = input_img.copy()
    draw = ImageDraw.Draw(image_with_box)
    # Draw the bounding box
    draw.rectangle(box_info, outline=box_color, width=box_width)

    return image_with_box


def outline_object(input_img, sam_info, outline_color='red', line_width=3):
    '''返回 对物体轮廓描边 的图像'''

    # 解码 segmentation mask
    mask = maskUtils.decode(sam_info['segmentation'])  # Binary mask (0: background, 1: object)
    
    # 找到轮廓坐标 (返回的是多个轮廓)
    contours = measure.find_contours(mask, 0.5)  # Threshold at 0.5 to get the boundary

    # 复制输入图像并创建绘图对象
    outlined_image = input_img.copy()
    draw = ImageDraw.Draw(outlined_image)

    # 绘制轮廓
    for contour in contours:
        contour = [(int(x), int(y)) for y, x in contour]  # 转换为整数坐标
        draw.line(contour, fill=outline_color, width=line_width)

    return outlined_image


def concatenate_images_horizontal(input_images, dist_images=10):
    '''横向拼接 2 张图片'''
    # calc total width of imgs + dist between them
    total_width = sum(img.width for img in input_images) + dist_images * (len(input_images) - 1)
    # calc max height from imgs
    height = max(img.height for img in input_images)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (total_width, height), (0, 0, 0))

    # init var to track current width pos
    current_width = 0
    for img in input_images:
        # paste img in new_img at current width
        new_img.paste(img, (current_width, 0))
        # update current width for next img
        current_width += img.width + dist_images

    return new_img


def turn_grayscale_image(input_img, manner='1'):
    '''图片转灰度图'''

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