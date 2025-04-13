import os
import json
import argparse
from tqdm import tqdm
from collections import Counter
from PIL import Image

import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from img_utli import box_object

# Prompt for color extraction
get_color_prompt = (
    "What are the main colors that appear in this image?\n"
    "List up to 5 common color names separated by commas.\n"
    "Example: red, blue, green\n"
    "Answer:"
)

# Color vocabulary (W3C-style)
COMMON_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "cyan", "magenta", "gold", "silver", "olive", "coral"
]

def qwen_generate(qwen_model, qwen_processor, img: Image.Image, prompt: str, temperature=0.3, top_p=0.9, max_new_tokens=32):
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(qwen_model.device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    return output_text.strip()

def extract_colors_from_text(output_text):
    output_text = output_text.lower()
    found_colors = [w.strip() for w in output_text.split(',')]
    matched = [c for c in found_colors if c in COMMON_COLORS]
    return list(set(matched))  # remove duplicates

def generate_confusers(gt_colors, total=3):
    pool = list(set(COMMON_COLORS) - set(gt_colors))
    return pool[:total] if len(pool) >= total else pool

def extract_colors_with_qwen(image_folder, save_path, qwen_model_path, topk=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_path, device_map=device)
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)

    image_list = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png'))])
    result_list = []

    for idx, img_name in enumerate(tqdm(image_list, desc="Extracting colors via Qwen")):
        img_path = os.path.join(image_folder, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            boxed_image = box_object(image, box=None)  # compatible with gray-box + red-box visualization

            output = qwen_generate(qwen_model, qwen_processor, boxed_image, get_color_prompt)
            gt_colors = extract_colors_from_text(output)

            color_choices = []
            for c in gt_colors:
                confuser = generate_confusers(gt_colors, total=3)
                color_choices.append({
                    "gt_color": c,
                    "confuser": confuser
                })

            result_list.append({
                "img_id": idx,
                "img_name": img_name,
                "gt_colors": gt_colors,
                "color_choices": color_choices,
                "raw_output": output
            })

        except Exception as e:
            print(f"[WARN] Failed to process {img_name}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result_list, f, indent=4)
    print(f"[INFO] Saved Qwen color extraction results to {save_path}")
    return result_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True, help="Path to input image folder")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the result JSON")
    parser.add_argument("--qwen_model_path", type=str, required=True, help="Path or HuggingFace name for Qwen-VL model")
    parser.add_argument("--topk", type=int, default=3, help="Number of top colors to extract")
    args = parser.parse_args()

    extract_colors_with_qwen(
        image_folder=args.image_folder,
        save_path=args.save_path,
        qwen_model_path=args.qwen_model_path,
        topk=args.topk
    )

if __name__ == "__main__":
    main()
