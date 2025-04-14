
import os
import json
import torch
import random
import argparse

from tqdm import tqdm
from infer_utili.data_utili import get_data, save_json
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor


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

def refine_ans(raw_answer):
    words = [word.strip().lower() for word in raw_answer.split(',') if word != '']
    return words

# Prompt for color extraction
get_color_prompt = (
    "Identify and list only the distinct colors visible in the image. "
    "Do not include descriptions, materials, or background details—only color names.\n"
    "Format your answer as a comma-separated list of colors. For example:\n"
    "red, blue, green\n"
    "Now, provide your answer:\n"
    "Answer:"
)

# Color vocabulary (W3C-style)
COMMON_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "cyan", "magenta", "gold", "silver", "olive", "coral"
]

rid_color = ['white', 'black', 'gray']
common_colors = [
    "red",      # 红色
    "blue",     # 蓝色
    "green",    # 绿色
    "yellow",   # 黄色
    "orange",   # 橙色
    "purple",   # 紫色
    "pink",     # 粉色
    "brown",    # 棕色
    # "black",    # 黑色
    # "white",    # 白色
    "cyan",     # 青色
    "magenta",  # 品红
    "gold",     # 金色
    "silver",   # 银色
    "olive",    # 橄榄绿
    "coral"     # 珊瑚色
    ]


def main(args):
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # get Qwen model
    qwen_model_add = "/data/yinjinhua/LMmodel/Qwen_Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_add, torch_dtype="auto", device_map=DEVICE)
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_add)

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # save add
    save_add = f'ImgColor_exp/confuser_res/{args.data_name}/confuser_res.json'
    os.makedirs(os.path.dirname(save_add), exist_ok=True)

    # start_pos = 0
    # end_pos = dataset_length
    res = []
    for sample_id in tqdm(range(dataset_length)):
        # 输出保存为 json, 其中图片用 id 指示
        current_img_id, current_img_label, current_image = sample_id, used_dataset['label'][sample_id], used_dataset[sample_id]['image']

        single_res = []
        # 对一张图片重复询问多次
        ask_time = 3
        for j in range(ask_time):
            raw_answer = qwen_inference(
                qwen_model=qwen_model, 
                qwen_processor=qwen_processor, 
                img=current_image, 
                prompt=get_color_prompt, 
                temperature=0.3, 
                top_p=0.9, 
                num_gen_token=64
                )
            refined_ans = refine_ans(raw_answer)
            single_res.extend(refined_ans)

        gt_colors = list(set(single_res))
        gt_colors = [color for color in gt_colors if color not in rid_color] # 图片提取得到的颜色中 去掉白、黑、灰
        potential_confuser_colors = [color for color in common_colors if color not in gt_colors]  # common_colors 里去掉 图片提取颜色  
        color_choices = []
        for c in gt_colors:
            confuser = random.sample(potential_confuser_colors, 3)
            color_choices.append({
                "gt_color": c,
                "confuser": confuser
            })

        res.append({
            'img_id': current_img_id,
            'ground_truth_label': current_img_label,
            'gt_colors': gt_colors,
            'color_choices': color_choices,
        })

        # save result
        save_json(save_add, res)
        

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
