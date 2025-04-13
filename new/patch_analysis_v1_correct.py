
import os
import json
import torch
import random
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from infer_utili.data_utili import get_data, save_json
from infer_utili.image_utils import reorder_image_patches

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


used_prompt_dict = {
    '1': "Write a caption describing the scene.",                       # 不会暴露 patch 信息
    '2': "Give a short description of what this image depicts.",        # 会暴露 patch 信息
    '3': "What is shown in this picture?",                              # 不会暴露 patch 信息
    '4': "Describe the contents of this photo.",                        # 会暴露 patch 信息
    '5': "Summarize the visual content in this image.",                 # 会暴露 patch 信息
}

def main(args):
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # get Qwen model
    qwen_model_add = "/data/yinjinhua/LMmodel/Qwen_Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_add, torch_dtype="auto", device_map=DEVICE)
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_add)

    dataset, dataset_length = get_data(args.data_name)

    start_pos = args.start_pos
    end_pos = args.end_pos
    save_dir = Path(f"/data/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/Patch_exp/v1_multiChoice/confuser_res/{args.data_name}/prompt_{args.desc_prompt_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'confuser_res_{start_pos}_end_{end_pos}.json'

    result_data = []

    for img_id in tqdm(range(start_pos, end_pos)):
        entry = dataset[img_id]
        label, image = entry['label'], entry['image']

        record = {
            "img_id": img_id,
            "ground_truth_label": label,
            "questions": []
        }

        gt_caption = qwen_inference(
            qwen_model=qwen_model,
            qwen_processor=qwen_processor,
            img=image,
            prompt=used_prompt_dict[str(args.desc_prompt_id)],
            temperature=0.6,
            top_p=0.9,
            num_gen_token=64
        )

        for reorder_idx in range(args.num_reorders):  # 对每张 original image 生成 3 个 reordered version
            reordered_img, perm = reorder_image_patches(image, n=3, perm=None) # 分成 3*3=9 patches
            confuser_captions = []
            for _ in range(3):  # 对每张 reorder version 生成 3 个描述作为 confuser
                cap = qwen_inference(
                    qwen_model=qwen_model,
                    qwen_processor=qwen_processor,
                    img=reordered_img,
                    prompt=used_prompt_dict[str(args.desc_prompt_id)],
                    temperature=0.6,
                    top_p=0.9,
                    num_gen_token=64
                )
                confuser_captions.append(cap)

            record["questions"].append({
                "perm": perm,
                "gt_caption": gt_caption,
                "confusers": confuser_captions
            })

        result_data.append(record)

    with open(save_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"[INFO] Saved output to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_reorders', type=int, default=3)
    parser.add_argument('--desc_prompt_id', type=int, default=1)
    parser.add_argument('--start_pos', type=int, default=0, help='start position of dataset')
    parser.add_argument('--end_pos', type=int, default=1, help='end position of dataset')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

# python patch_analysis_v1_correct.py --data_name img_Flickr --gpu_id 3 --start_pos 0 --end_pos 300



