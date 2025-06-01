import os
import re
import json
import time
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

from infer_utili.logging_func import *
from infer_utili.image_cache import ImageCache
from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json, format_elapsed_time
from infer_utili.llava_infer import llava_oneSample_inference
from infer_utili.llama_adapter_infer import llama_adapter_oneSample_inference
from infer_utili.minigpt4_infer import minigpt4_oneSample_inference


INFERENCE_FUNCS_sample = {
    'llava_v1_5_7b': llava_oneSample_inference,
    'MiniGPT4': minigpt4_oneSample_inference,
    'llama_adapter_v2': llama_adapter_oneSample_inference,
}

def clip_similarity(model, processor, img_patch, text):

    inputs = processor(text=text, images=img_patch, return_tensors="pt", padding=True).to(model.device)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # (1, 768)
        text_embeds = outputs.text_embeds    # (1, 768)

    # Compute cosine similarity
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    similarity = (image_embeds @ text_embeds.T).item()

    return similarity


def main(args):

    # Load CLIP model 
    DEVICE = torch.device(f'cuda:{args.gpu_id}')
    clip_model = CLIPModel.from_pretrained("openai_clip-vit-large-patch14-336")
    clip_processor = CLIPProcessor.from_pretrained("openai_clip-vit-large-patch14-336")
    clip_model.eval().to(DEVICE)

    # get dataset
    dataset, dataset_length = get_data(args.data_name)

    # get target model
    print("[INFO] Loading model...")
    model_dict = load_target_model(args)
    model_dict["image_cache"] = ImageCache(max_size=32)
    inference_func = INFERENCE_FUNCS_sample[args.target_model]

    prompt = 'Describe this image in detail.'

    # get confuser result
    confuser_add = f'ObjColor_exp/confuser_res/{args.filter_apply}/{args.data_name}_by_gpt-4o-mini/res.json'
    confuser_res = read_json(confuser_add)

    # calculate clip for every image patch 
    for sample_id in tqdm(range(dataset_length)):   

        image = dataset[sample_id]['image']
        # get image description from target model
        desc_text = inference_func(model_dict, prompt, image, args)[0]

        obj_info = confuser_res[sample_id]['sam_result']
        if obj_info is not None:
            for obj_sam_res in obj_info:
                box_info = obj_sam_res['bbox']
                # obtain image patch based on box 
                img_patch = image.crop(box_info)
                clip_score = clip_similarity(clip_model, clip_processor, img_patch, desc_text)
                obj_sam_res['clip_score'] = clip_score

    # save updated result (with clip score)
    save_dir = Path(f'ObjColor_exp/confuser_res/{args.filter_apply}/{args.data_name}_by_gpt-4o-mini/{args.target_model}')
    save_dir.mkdir(parents=True, exist_ok=True)
    print(save_dir)

    save_path = save_dir / f'res_w_clip.json'
    save_json(save_path, confuser_res)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--target_model', type=str, default='llava_v1_5_7b')
    parser.add_argument('--filter_apply', type=str, choices=['noFilter', 'withFilter'], required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)