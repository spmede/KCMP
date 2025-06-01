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
from infer_utili.apiCallFunc import apiCall_img, apiCall
from infer_utili.data_utili import get_data, read_json, save_json, format_elapsed_time


def regenerate_single_obj_confuser(input_img, single_sam_res, args, maksed_img_desc, existing_confusers, gt_obj_name, max_attempts=5):
    """
    Regenerate a single object confuser until it is plausible and unique.
    """
    for attempt in range(max_attempts):
        new_confusers = get_obj_name_confuser(input_img, single_sam_res, args)
        if not new_confusers:
            continue
        for new_confuser in new_confusers:
            if new_confuser == gt_obj_name or new_confuser in existing_confusers:
                continue  # skip duplicates
            plausibility = apiCall(
                prompt=build_obj_confuser_plausibility_check_prompt(maksed_img_desc, new_confuser),
                num_generation=1,
                temp=args.temperature,
                model=args.llm_APImodel,
            )[0].lower()
            if plausibility == 'yes':
                return new_confuser, plausibility
    return None, None  # failed


def regenerate_single_color_confuser(class_name, args, existing_confusers, gt_color, max_attempts=5):
    """
    Regenerate a single color confuser until plausible and unique.
    """
    available_colors = [c for c in common_colors if c != gt_color and c not in existing_confusers]
    for attempt in range(max_attempts):
        if not available_colors:
            return None, None
        new_color = random.choice(available_colors)
        plausibility = apiCall(
            prompt=build_color_confuser_plausibility_check_prompt(class_name, new_color),
            num_generation=1,
            temp=args.temperature,
            model=args.llm_APImodel,
        )[0].lower()
        if plausibility == 'yes':
            return new_color, plausibility
        else:
            available_colors.remove(new_color)  # remove implausible
    return None, None


masked_desc_prompt = (
    "Task:\n"
    "You are given an image where an object is covered by a black rectangle.\n"
    "Generate a natural and fluent description of the visible parts of the scene.\n\n"
    "Requirements:\n"
    "- Use [MASK] to naturally represent the missing object.\n"
    "- Treat [MASK] as if it were an ordinary, visible part of the scene.\n"
    "- Do not mention 'black rectangle', 'mask', 'covered', 'hidden', or anything related to masking.\n"
    "- Do not guess or invent details about what [MASK] might be.\n"
    "- Write only objective, factual observations. Do not add any emotional, aesthetic, or subjective language (e.g., 'beautiful').\n"
    "- Write 1-2 complete and fluent sentences.\n\n"
    "Examples:\n"
    "- \"A branch with several buds extends horizontally, with a [MASK] positioned near the center.\"\n"
    "- \"Three ceramic bowls of different sizes are arranged on clear stands, with a [MASK] between them.\"\n"
    "- \"A table holds a laptop, a notebook, and a [MASK] placed next to each other.\"\n\n"
    )


def build_obj_gt_certainty_check_prompt(masked_image_description, groundtruth_object):
    prompt = (
        "Task:\n"
        "You are given a description of an image where a part is masked (shown as [MASK]).\n"
        "Determine how certain it is that [MASK] corresponds to the given object.\n"
        "Answer with exactly one of: Very Certain / Possible / Unlikely.\n\n"
        "Examples:\n"
        "- Description: \"A woman is walking her [MASK] on a leash along the sidewalk.\"\n"
        "  Object: \"dog\" → Answer: Very Certain\n"
        "- Description: \"A [MASK] is sitting next to a coffee cup on the table.\"\n"
        "  Object: \"book\" → Answer: Possible\n"
        "- Description: \"A beautiful landscape shows mountains, rivers, and a [MASK] in the distance.\"\n"
        "  Object: \"apple\" → Answer: Unlikely\n\n"
        "Now judge:\n"
        f"- Description: \"{masked_image_description}\"\n"
        f"- Object: \"{groundtruth_object}\"\n"
        "Answer:"
    )
    return prompt


def build_obj_confuser_plausibility_check_prompt(masked_image_description, confuser_object):
    prompt = (
        "Task:\n"
        "You are given a description of an image where a small part is masked (represented as [MASK]).\n"
        "Determine if the given object could plausibly fit into the masked region.\n"
        "Answer with exactly one word: Yes / No.\n\n"
        "Examples:\n"
        "- Description: \"A man is holding a [MASK] while standing on a sports field.\"\n"
        "  Object: \"soccer ball\" → Answer: Yes\n"
        "- Description: \"A [MASK] is sitting on a windowsill covered in flowers.\"\n"
        "  Object: \"laptop\" → Answer: No\n"
        "- Description: \"A group of people are having a picnic around a [MASK].\"\n"
        "  Object: \"tree\" → Answer: Yes\n\n"
        "Now judge:\n"
        f"- Description: \"{masked_image_description}\"\n"
        f"- Object: \"{confuser_object}\"\n"
        "Answer:"
    )
    return prompt


def build_color_gt_certainty_check_prompt(object_name, groundtruth_color):
    prompt = (
        "Task:\n"
        "You are given the name of an object and a proposed primary color for that object in an image.\n"
        "Your job is to determine how certain it is that the object in the image has this color.\n"
        "Answer with exactly one of: Very Certain / Possible / Unlikely.\n\n"
        "Examples:\n"
        "- Object: \"apple\" — Color: \"red\" → Answer: Very Certain\n"
        "- Object: \"car\" — Color: \"blue\" → Answer: Possible\n"
        "- Object: \"dog\" — Color: \"purple\" → Answer: Unlikely\n\n"
        "Now judge:\n"
        f"- Object: \"{object_name}\" — Color: \"{groundtruth_color}\"\n"
        "Answer:"
    )
    return prompt


def build_color_confuser_plausibility_check_prompt(object_name, confuser_color):
    prompt = (
    "Task:\n"
    "You are given the name of an object and a color.\n"
    "Your job is to determine whether this color is natural, common, or plausible for this object in the real world.\n"
    "If the color is reasonable, answer 'Yes'. If unnatural or highly unlikely, answer 'No'.\n"
    "Only output 'Yes' or 'No' without explanation.\n\n"
    "Examples:\n"
    "- Object: \"apple\" — Color: \"red\" → Yes\n"
    "- Object: \"sky\" — Color: \"green\" → No\n"
    "- Object: \"rose\" — Color: \"blue\" → No\n"
    "- Object: \"banana\" — Color: \"yellow\" → Yes\n"
    "- Object: \"cat\" — Color: \"purple\" → No\n"
    "- Object: \"car\" — Color: \"orange\" → Yes\n\n"
    "Now judge:\n"
    f"- Object: \"{object_name}\" — Color: \"{confuser_color}\""
    )
    return prompt


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


def main(args):

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # get confuser res 
    confuser_res = read_json(f'ObjColor_exp/confuser_res/noFilter/{args.data_name}_by_gpt-4o-mini_mode_default/res.json')
    
    # save add
    save_dir = Path(f"ObjColor_exp/confuser_res/withFilter/{args.data_name}_by_{args.vllm_APImodel}/{time.strftime('%Y%m%d-%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    # initialize logger
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    start_pos, end_pos = 0, dataset_length
    save_path = save_dir / f'res_start_{start_pos}_end_{end_pos}.json'

    # timing
    start_time = time.time()

    # check and update confuser
    updated_confuser_res = []

    for i in tqdm(range(start_pos, end_pos)):

        sam_result = confuser_res[i]['sam_result']
        original_img_id = confuser_res[i]['original_img_id']

        if sam_result:
            for obj in sam_result:
                class_name = obj['class_name']

                # 1. object processing
                obj_name_distractors = obj['obj_name_distractors']
                if obj_name_distractors: # not None

                    maksed_img = mask_object(used_dataset[original_img_id]['image'], obj)
                    # get masked desc
                    maksed_img_desc = apiCall_img(
                        maksed_img, 
                        masked_desc_prompt, 
                        temperature=args.temperature,
                        num_gen_token=128, 
                        model=args.vllm_APImodel, 
                        )
                
                    # determine the quality of groundtruth answer from object query
                    gt_obj_resonable = apiCall(
                        prompt=build_obj_gt_certainty_check_prompt(maksed_img_desc, class_name), 
                        num_generation=1,
                        temp=args.temperature,
                        model=args.llm_APImodel,
                        )[0].lower()  # Very Certain / Possible / Unlikely
                    
                    if gt_obj_resonable == 'very certain':
                        obj['ask_masked_obj_validity'] = False
                    else:
                        obj['ask_masked_obj_validity'] = True
                        # determine the quality of confuser options from object query
                        confuser_obj_resonable = [
                            apiCall(
                                prompt=build_obj_confuser_plausibility_check_prompt(maksed_img_desc, obj_confuser), 
                                num_generation=1,
                                temp=args.temperature,
                                model=args.llm_APImodel, 
                                )[0].lower() for obj_confuser in obj_name_distractors
                                ]
                        # regenerate invalid confuser option 
                        for idx, res in enumerate(confuser_obj_resonable):
                            if res != 'yes':
                                new_confuser, new_plausibility = regenerate_single_obj_confuser(
                                    used_dataset[original_img_id]['image'],
                                    obj,
                                    args,
                                    maksed_img_desc,
                                    existing_confusers=[class_name] + [obj_name_distractors[j] for j in range(len(obj_name_distractors)) if confuser_obj_resonable[j] == 'yes'],
                                    gt_obj_name=class_name
                                )
                                if new_confuser:
                                    obj_name_distractors[idx] = new_confuser
                                    confuser_obj_resonable[idx] = new_plausibility
                        # save final
                        obj['obj_name_distractors'] = obj_name_distractors
                        # recore quality check, e.g., obj['ask_masked_obj_confuser_possibility'] = ['yes', 'no', 'yes']
                        obj['ask_masked_obj_confuser_possibility'] = confuser_obj_resonable
                        
                # 2. color processing
                obj_gt_color, obj_gt_color_distractors = obj['obj_gt_color'], obj['obj_gt_color_distractors']
                if obj_gt_color: # not None
                    
                    # determine the quality of groundtruth answer from color query
                    gt_color_resonable = apiCall(
                        prompt=build_color_gt_certainty_check_prompt(class_name, obj_gt_color), 
                        num_generation=1,
                        temp=args.temperature,
                        model=args.llm_APImodel)[0].lower()  # Very Certain / Possible / Unlikely
                    
                    if gt_color_resonable == 'very certain':
                        obj['ask_obj_color_validity'] = False
                    else:
                        obj['ask_obj_color_validity'] = True

                        # determine the quality of confuser options from color query
                        confuser_color_resonable  = [
                            apiCall(
                                prompt=build_color_confuser_plausibility_check_prompt(class_name, color), 
                                num_generation=1,
                                temp=args.temperature,
                                model=args.llm_APImodel)[0].lower() for color in obj_gt_color_distractors
                                ]
                        # regenerate invalid confuser option 
                        for idx, res in enumerate(confuser_color_resonable):
                            if res != 'yes':
                                new_color, new_plausibility = regenerate_single_color_confuser(
                                    class_name,
                                    args,
                                    existing_confusers=[obj_gt_color] + [obj_gt_color_distractors[j] for j in range(len(obj_gt_color_distractors)) if confuser_color_resonable[j] == 'yes'],
                                    gt_color=obj_gt_color
                                )
                                if new_color:
                                    obj_gt_color_distractors[idx] = new_color
                                    confuser_color_resonable[idx] = new_plausibility
                        # save final
                        obj['obj_gt_color_distractors'] = obj_gt_color_distractors
                        # recore quality check, e.g., obj['ask_obj_color_confuser_possibility'] = ['yes', 'yes', 'no']
                        obj['ask_obj_color_confuser_possibility'] = confuser_color_resonable

                       
        updated_confuser_res.append(confuser_res[i])
        # save result
        save_json(save_path, updated_confuser_res)

    # record time
    time_notice = format_elapsed_time(start_time)
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr', 
                        choices=['img_Flickr', 'img_Flickr_10k', 'img_Flickr_2k', 'img_dalle'])
    
    parser.add_argument('--vllm_APImodel', type=str, default='gpt-4o-mini')
    parser.add_argument('--llm_APImodel', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--temperature', type=float, default=0.3)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parse()
    main(args)