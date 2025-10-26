import os
import re
import time
import json
import random
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from infer_utili.image_utils import *
from infer_utili.logging_func import *
from infer_utili.apiCallFunc import apiCall_img
from infer_utili.data_utili import get_data, read_json, save_json, format_elapsed_time


# used prompt for object confuser generation
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


def get_obj_name_confuser(input_img, single_sam_res, args, max_attempts=5):
    attempt = 0
    distractors = set()

    gt_obj_name = single_sam_res['class_name']

    masked_image = mask_object(input_img, single_sam_res)
    concatenated_image = concatenate_images_horizontal(input_images=[input_img, masked_image], dist_images=20)

    while attempt < max_attempts:
        ans = apiCall_img(
            concatenated_image, 
            combined_img_obj_confuser_prompt.format(gt_obj_name), 
            temperature=0.6,
            num_gen_token=64,
            model=args.vllm_APImodel, 
        )

        answers = re.findall(r"\*(.*?)\*", ans)

        if answers:  # at least one answer is extracted
            different_answers = [a.lower() for a in answers if a.lower() != gt_obj_name.lower()]
            distractors.update(different_answers)
            if len(distractors) >= 3:
                return random.sample(list(distractors), 3)  # Return as soon as we get 3 different ones

        attempt += 1
    return None


# Filter out large-area objects that contain more than one primary color
get_color_prompt = (
    "What is the primary color of the object enclosed in the red box in the image?\n"
    "If the object contains multiple dominant colors, answer with *MULTICOLOR*.\n"
    "Otherwise, format your color answer between '*'. For example, *Black*.\n"
    "Answer:"
    )

common_colors = [
    "red",     
    "blue",    
    "green",   
    "yellow",   
    "orange",   
    "purple",   
    "pink",  
    "brown",  
    "black", 
    "white",   
    "cyan",    
    "magenta",  
    "gold",     
    "silver",   
    "olive",   
    "coral"  
    ]


def get_true_obj_color(input_img, single_sam_res, args, ask_time=5, max_retries=3):
    """
    Query the primary color of the boxed object in the image.
    If the object is multi-colored (detected by the model), will return None.
    """
    image_with_box = box_object(input_img, single_sam_res, box_color='red', box_width=3)

    for retry in range(max_retries):
        potential_colors = []

        for _ in range(ask_time):
            response = apiCall_img(
                image_with_box,
                get_color_prompt,
                temperature=0.6,
                num_gen_token=64,
                model=args.vllm_APImodel,
            )

            extracted = re.findall(r"\*(.*?)\*", response)
            if extracted:
                color = extracted[0].strip().lower()
                # Check if model reported multi-color
                if color == "multicolor":
                    # print("Model indicated MULTICOLOR object; skipping.")
                    return None, []  # immediately return, avoid invalid color
                potential_colors.append(color)

        valid_colors = [c for c in potential_colors]

        if valid_colors:
            true_color = Counter(valid_colors).most_common(1)[0][0]
            break
    else:
        true_color = None  # retries exhausted

    if true_color is None:
        confuser_colors = []
    else:
        confuser_colors = random.sample([c for c in common_colors if c != true_color], 3)

    return true_color, confuser_colors


def main(args):

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # read saved image analysis, e.g., vllm_APImodel is 'gpt-4o-mini' in 'ObjColor_image_analysis.py'
    result = read_json(f'ObjColor_exp/img_analysis_res/{args.data_name}_by_gpt-4o-mini/res.json')
    
    # save add
    save_dir = Path(f"ObjColor_exp/confuser_res/noFilter/{args.data_name}_by_{args.vllm_APImodel}")
    save_dir.mkdir(parents=True, exist_ok=True)

    start_pos, end_pos = 0, dataset_length
    save_path = save_dir / f'res.json'

    # initialize logger and record all parameters
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    # timing
    start_time = time.time()

    # confuser generation
    confuser_result = []

    for sample_id in tqdm(range(start_pos, end_pos)):

        # save res in json format
        current_img_id, current_img_label, current_image = sample_id, used_dataset['label'][sample_id], used_dataset[sample_id]['image']
        current_sam_list = result[sample_id]['sam_result'] # keys: ['original_img_id', 'ground_truth_label', 'object_name', 'sam_result']
        
        current_img_dict = result[sample_id].copy()

        if current_sam_list:
            # Create a new list to store the updated sam results
            updated_sam_list = []

            # for single_sam_res in tqdm(current_sam_list):
            for single_sam_res in current_sam_list: 

                # get object name confuser
                obj_name_distractors = get_obj_name_confuser(
                    input_img=current_image,
                    single_sam_res=single_sam_res,
                    args=args,
                    max_attempts=5
                    )
                
                # get ground truth color confuser
                obj_gt_color, obj_gt_color_distractors = get_true_obj_color(
                    input_img=current_image,
                    single_sam_res=single_sam_res,
                    args=args,
                    ask_time=5,
                    max_retries=5
                    )

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
        save_json(save_path, confuser_result)

    # record time
    time_notice = format_elapsed_time(start_time)
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr', 
                        choices=['img_Flickr', 'img_Flickr_10k', 'img_Flickr_2k', 'img_dalle'])
    
    parser.add_argument('--vllm_APImodel', type=str, default='gpt-4o-mini')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse()
    main(args)