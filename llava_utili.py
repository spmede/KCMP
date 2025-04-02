
'''
llava 作为 target model
'''

import re
import torch
import requests
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
paths_to_add = [
    Path("/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA"),
]
for custom_path in paths_to_add:
    if str(custom_path) not in sys.path:
        sys.path.append(str(custom_path)) 

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.conversation import conv_templates
from llava.model import *
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path



# -------------- 图像加载 相关函数
def load_image_llava(image_file):
    '''加载 本地/网络图像 并转为RGB格式'''
    if isinstance(image_file, Image.Image):  
        return image_file.convert("RGB")  
    
    if isinstance(image_file, str) and (image_file.startswith("http") or image_file.startswith("https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image_llava(image_file)
        out.append(image)
    return out


# -------------- LLaVA 相关函数
def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode


# --------- llava 推理
def llava_inference(model_dict, input_query, img, args, do_sample=True):
    
    llava_model = model_dict["model"] 
    llava_tokenizer = model_dict["tokenizer"]
    image_processor = model_dict["image_processor"] 
    conv_mode = model_dict["conv_mode"] 

    qs = input_query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    # 构建提示文本
    if IMAGE_PLACEHOLDER in qs:
        if llava_model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if llava_model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # 创建对话模板
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 处理图像
    images = load_images([img])
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        llava_model.config
    ).to(llava_model.device, dtype=torch.float16)

    input_ids, prompt_chunks = tokenizer_image_token(prompt, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda(args.gpu_id)

    # 生成文本
    with torch.inference_mode():
        output_ids = llava_model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=args.temperature, 
            top_p=args.top_p, 
            max_new_tokens=args.num_gen_token,
            # use_cache=True,
        )

    output_text = llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return output_text


