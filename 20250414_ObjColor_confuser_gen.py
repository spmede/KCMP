
import os
import re
import json
import torch
import random
import argparse

from tqdm import tqdm
from collections import Counter
from infer_utili.data_utili import get_data, read_json, save_json
from infer_utili.image_utils import *

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


# ask_obj 生成混淆选项, 提供原图以及 mask 图, 由 Qwen 生成可能的物体名称
combined_img_obj_confuser_prompt = (
    "You are given two images:\n"
    "1. The original image.\n"
    "2. The same image with a region masked out.\n\n"

    "Additional context:\n"
    "The masked object is a '{}' in the original image.\n\n"

    "Your task:\n"
    "1. Look at both the original image and the masked image.\n"
    "2. Based on the surrounding context, propose 5 different objects that could realistically appear in the masked region.\n"
    "3. Format each object between '*'. For example, *Dog*.\n"
    "4. Do not generate any explanations.\n\n"

    "Answer:"
)

# 询问框中物体的颜色
get_color_prompt = (
    'What is the primary color of the object enclosed in the red box in the image?\n'
    "Format your answer between '*'. For example, *Black*.\n"
    'Answer:'
    )

# Qwen inference function
def qwen_inference(qwen_model, qwen_processor, img, prompt, temperature, top_p, num_gen_token):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", 
                 "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)

    # Inference: Generation of the output
    generated_ids = qwen_model.generate(**inputs, 
                                        max_new_tokens=num_gen_token,
                                        temperature=temperature,
                                        top_p=top_p)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def get_obj_name_confuser(qwen_model, qwen_processor, input_img, single_sam_res, max_attempts=10):
    attempt = 0
    distractors = set()

    gt_obj_name = single_sam_res['class_name']

    masked_image = mask_object(input_img, single_sam_res)
    concatenated_image = concatenate_images_horizontal(input_images=[input_img, masked_image], dist_images=20)

    while attempt < max_attempts:
        ans = qwen_inference(
            qwen_model=qwen_model, 
            qwen_processor=qwen_processor, 
            img=concatenated_image, 
            prompt=combined_img_obj_confuser_prompt.format(gt_obj_name), 
            temperature=0.6, 
            top_p=0.9, 
            num_gen_token=64)

        answers = re.findall(r"\*(.*?)\*", ans)

        if answers:  # at least one answer is extracted
            different_answers = [a for a in answers if a != gt_obj_name]
            distractors.update(different_answers)
            if len(distractors) >= 3:
                return random.sample(list(distractors), 3)  # Return as soon as we get 3 different ones

        attempt += 1
    return None


common_colors = [
    "red",      # 红色
    "blue",     # 蓝色
    "green",    # 绿色
    "yellow",   # 黄色
    "orange",   # 橙色
    "purple",   # 紫色
    "pink",     # 粉色
    "brown",    # 棕色
    "black",    # 黑色
    "white",    # 白色
    "cyan",     # 青色
    "magenta",  # 品红
    "gold",     # 金色
    "silver",   # 银色
    "olive",    # 橄榄绿
    "coral"     # 珊瑚色
    ]

def get_true_obj_color(qwen_model, qwen_processor, input_img, single_sam_res, ask_time=5, max_retries=3):
    
    image_with_box = box_object(input_img, single_sam_res, box_color='red', box_width=3)

    for retry in range(max_retries):
        true_color_list = []

        for _ in range(ask_time):
            potential_color = qwen_inference(
                                qwen_model=qwen_model, 
                                qwen_processor=qwen_processor, 
                                img=image_with_box, 
                                prompt=get_color_prompt,
                                temperature=0.3, 
                                top_p=0.9, 
                                num_gen_token=8
                                )
            extracted_colors = re.findall(r"\*(.*?)\*", potential_color)
            if extracted_colors:  # Ensure there's at least one match
                potential_color_extract = extracted_colors[0].lower()
                true_color_list.append(potential_color_extract)

        # Filter colors that exist in `common_colors`  限制了颜色只能是 common colors 里的; 这个限制可以解除, 不限制模型生成的 gt color 
        refined_color_list = [color for color in true_color_list if color in common_colors]

        if refined_color_list:
            true_color = Counter(refined_color_list).most_common(1)[0][0]
            break  # Exit retry loop if we found a valid color
        # else:  # 控制是否输出提示文字
        #     print(f"Retry {retry + 1}/{max_retries}: No valid colors found, trying again...")   

    else: 
        # If all retries fail, use None for true_color
        # print("Max retries reached. True color set as None.")
        true_color = None

    if true_color is None:
        confuser_colors = []
    else:
        # Select 3 confuser colors, ensuring they are different from true_color
        confuser_colors = random.sample([color for color in common_colors if color != true_color], 3)
    
    return true_color, confuser_colors


def main(args):
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # get Qwen model
    qwen_model_add = "/data/yinjinhua/LMmodel/Qwen_Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_add, torch_dtype="auto", device_map=DEVICE)
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_add)

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # 读取 image analysis 结果
    result = read_json(f'ObjColor_exp/img_analysis_res/{args.data_name}/analysis_res.json')
    
    # save add
    save_add = f'ObjColor_exp/confuser_res/{args.data_name}/confuser_res.json'
    os.makedirs(os.path.dirname(save_add), exist_ok=True)

    # confuser generation
    confuser_result = []

    for sample_id in tqdm(range(dataset_length)):
        # 输出保存为 json, 其中图片用 id 指示
        current_img_id, current_img_label, current_image = sample_id, used_dataset['label'][sample_id], used_dataset[sample_id]['image']
        current_sam_list = result[sample_id]['sam_result'] # keys: ['original_img_id', 'ground_truth_label', 'object_name', 'sam_result']
        
        current_img_dict = result[sample_id].copy()

        if current_sam_list:
            # Create a new list to store the updated sam results
            updated_sam_list = []

            # for single_sam_res in tqdm(current_sam_list):
            for single_sam_res in current_sam_list: 

                # get object name confuser
                obj_name_distractors = get_obj_name_confuser(qwen_model=qwen_model, 
                                                             qwen_processor=qwen_processor,
                                                             input_img=current_image,
                                                             single_sam_res=single_sam_res,
                                                             max_attempts=10)
                
                # get ground truth color confuser
                obj_gt_color, obj_gt_color_distractors = get_true_obj_color(qwen_model=qwen_model, 
                                                                            qwen_processor=qwen_processor,
                                                                            input_img=current_image,
                                                                            single_sam_res=single_sam_res,
                                                                            ask_time=5,
                                                                            max_retries=3)

                # Add the obj_name_distractors to the single_sam_res
                single_sam_res['obj_name_distractors'] = obj_name_distractors
                single_sam_res['obj_gt_color'] = obj_gt_color
                single_sam_res['obj_gt_color_distractors'] = obj_gt_color_distractors

                # Add the updated single_sam_res to the updated list
                updated_sam_list.append(single_sam_res)
            
            # Update the sam_result in current_img_dict with the updated list
            current_img_dict['sam_result'] = updated_sam_list

        confuser_result.append(current_img_dict)

        # save result
        save_json(save_add, confuser_result)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr', 
                        choices=['img_Flickr', 'img_Flickr_10k', 'img_Flickr_2k', 
                                 'img_dalle'])
    
    parser.add_argument('--gpu_id', default=1, type=int, help='visible gpu ids')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    main(args)


