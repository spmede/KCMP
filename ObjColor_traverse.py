import os
import re
import json
import time
import argparse
from math import ceil
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from collections import defaultdict

from eval_auc import *
from infer_utili.logging_func import *
from infer_utili.image_cache import ImageCache
from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json, format_elapsed_time
from infer_utili.image_utils import mask_object, box_object, turn_grayscale_image
from infer_utili import llava_batch_inference, minigpt4_batch_inference, llama_adapter_batch_inference


# we provide diverse queries and you can use any combination
PROMPT_TEMPLATES = {
    'ask_masked_obj': {
        '1': (
            "You have seen the image in your training data. Choose the option that correctly identifies the original content of the masked area.\n"
            "Options:\n{}\n{}\n{}\n{}\nAnswer:"
        ),
        '2': (
            "You have seen this image in your training data. Identify the original content of the masked area.\n"
            "Select exactly one option from the list below and respond with only that option. Do not add any extra words, phrases, or formatting.\n"
            "Your answer must be exactly one of the following:\n{}\n{}\n{}\n{}\n"
            "Strictly follow this format:\n"
            "Answer: <chosen_option>\n"
            "Example output:\n"
            "Answer: book\n"
            "Now, provide your answer:\n"
            "Answer:"
        ),
        '3': (
            "You are given an image with one region masked (black box). "
            "However, you have seen the original version of this image during training.\n\n"
            "Your task: Identify the original object that was masked.\n\n"
            "Choose ONLY ONE of the following options. Do NOT explain. Do NOT add extra text.\n"
            "Respond with ONLY the object name.\n\n"
            "Options:\n{}\n{}\n{}\n{}\n\n"
            "Format your answer exactly like this:\nAnswer: [object name]\n\n"
            "Example:\nAnswer: backpack\n\n"
            "Now provide your answer:\nAnswer:"
        ),

    },
    'ask_obj_color': {
        '1': (
            "The image is in grayscale, but you have encountered it during training.\n"
            "Identify the original color of the object enclosed in the red box.\n"
            "Options:\n{}\n{}\n{}\n{}\nAnswer:"
        ),
        '2': (
            "The image is in grayscale, but you have encountered it during training. "
            "Identify the original color of the object enclosed in the red box.\n"
            "You must choose one option. Do not generate any explanations.\n"
            "Options:\n{}\n{}\n{}\n{}\nAnswer:"
        ),
        '3': (
            "You are given a grayscale image. However, you have seen the original version during training.\n"
            "Your task: Identify the original color of the object enclosed in the red box.\n\n"
            "You must choose ONLY ONE of the following options. Do NOT explain. Do NOT add extra text.\n"
            "Respond with ONLY the color name.\n\n"
            "Options:\n{}\n{}\n{}\n{}\n\n"
            "Format your answer exactly like this:\nAnswer: [color]\n\n"
            "Example:\nAnswer: silver\n\n"
            "Now provide your answer:\nAnswer:"
        ),
    }
}

INFERENCE_FUNCS = {
    'llava_v1_5_7b': llava_batch_inference,
    'MiniGPT4': minigpt4_batch_inference,
    'llama_adapter_v2': llama_adapter_batch_inference,
}


def safe_extract_answer(text, options):
    text_clean = text.strip().lower()
    options_clean = [opt.lower() for opt in options]

    # Exact match
    for opt in options_clean:
        if text_clean == opt:
            return opt
        
    # Answer pattern: first non-empty line exact match  (suitable for MiniGPT4 answer)
    first_line = next((line.strip().lower() for line in text.splitlines() if line.strip()), '')
    if first_line in options_clean:
        return first_line
        
    # pattern like "color is <color>"
    match = re.search(r"color.*is\s+(.+)", text_clean)
    if match:
        candidate_phrase = match.group(1).strip().lower().rstrip(".")
        tokens = re.split(r"[\s,]+", candidate_phrase)
        matched = [opt for opt in options_clean if opt in tokens]
        if len(matched) == 1:
            return matched[0]
        elif len(matched) > 1:
            return None  # ambiguous → reject

    # Longest option fully contained in text
    matched = [opt for opt in options_clean if opt in text_clean]
    if len(matched) == 1:
        return matched[0]
    elif len(matched) > 1:
        return None  # ambiguous → reject

    # Try extracting known answer patterns
    match = re.search(r"answer\s*[::\-]?\s*(.+)", text, re.IGNORECASE)
    if match:
        response = match.group(1).strip().lower()
        for opt in options_clean:
            if response == opt:
                return opt

    # Fallback: substring word boundary matching
    for opt in options_clean:
        pattern = r'\b' + re.escape(opt) + r'\b'
        if re.search(pattern, text_clean):
            return opt

    return None


def prepare_queries(entry, ask_type, ask_time, ask_masked_obj_prompt_id, ask_obj_color_prompt_id):
    """
    Generate queries for both ask_masked_obj and ask_obj_color types.
    Each object yields two separate questions (object name, object color).
    """
    queries = []
    img_id = entry["original_img_id"]
    sam_result = entry.get("sam_result", None)
    if not sam_result:
        return queries  # no object detected → skip image

    for sam_id, sam in enumerate(entry["sam_result"]):
        # === 1. ask_masked_obj ===
        distractors = sam.get("obj_name_distractors")
        if distractors and sam['ask_masked_obj_validity']:  # 3 confusers exist and question is valid
            gt = sam.get("class_name", None)
            options = [gt] + distractors
            if ask_type == "ordered_choice":
                for i in range(4):
                    reordered = options[:]
                    reordered[0], reordered[i] = reordered[i], reordered[0]
                    prompt = PROMPT_TEMPLATES['ask_masked_obj'][str(ask_masked_obj_prompt_id)].format(*reordered)
                    queries.append({
                        'img_id': img_id,
                        'question_type': 'ask_masked_obj',
                        'object_index': sam_id, 
                        'sam': sam,
                        'prompt': prompt,
                        'options': reordered,
                        'gt': gt,
                        'max_retries': 5
                    })
            else:
                for _ in range(ask_time):
                    randomized = options[:]
                    shuffle(randomized)
                    prompt = PROMPT_TEMPLATES['ask_masked_obj'][str(ask_masked_obj_prompt_id)].format(*randomized)
                    queries.append({
                        'img_id': img_id,
                        'question_type': 'ask_masked_obj',
                        'object_index': sam_id,
                        'sam': sam,
                        'prompt': prompt,
                        'options': randomized,
                        'gt': gt,
                        'max_retries': 4 
                    })
        
        # === 2. ask_obj_color ===
        gt_color = sam.get("obj_gt_color", None)
        if gt_color and sam['ask_obj_color_validity']:  # 3 confusers exist and question is valid
            color_distractors = sam.get("obj_gt_color_distractors")
            color_options = [gt_color] + color_distractors
            if ask_type == "ordered_choice":
                for i in range(4):
                    reordered = color_options[:]
                    reordered[0], reordered[i] = reordered[i], reordered[0]
                    prompt = PROMPT_TEMPLATES['ask_obj_color'][str(ask_obj_color_prompt_id)].format(*reordered)
                    queries.append({
                        'img_id': img_id,
                        'question_type': 'ask_obj_color',
                        'object_index': sam_id, 
                        'sam': sam,
                        'prompt': prompt,
                        'options': reordered,
                        'gt': gt_color,
                        'max_retries': 5
                    })
            else:
                for _ in range(ask_time):
                    randomized = color_options[:]
                    shuffle(randomized)
                    prompt = PROMPT_TEMPLATES['ask_obj_color'][str(ask_obj_color_prompt_id)].format(*randomized)
                    queries.append({
                        'img_id': img_id,
                        'question_type': 'ask_obj_color',
                        'object_index': sam_id,
                        'sam': sam,
                        'prompt': prompt,
                        'options': randomized,
                        'gt': gt_color,
                        'max_retries': 4 
                    })
    
    return queries


def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    """
    For each query: attempt multiple retries to get a valid answer.
    Return:
        raw_ans_list: list[list[str]]  all generations for each query
        acc_list: list[int]  per-query accuracy (0 or 1)
        invalid_info_list: list[int]  0 = valid, 1 = invalid
    """
    total_queries = len(queries)
    raw_ans_list = [[] for _ in range(total_queries)]
    acc_list = [0] * total_queries
    invalid_info_list = [1] * total_queries
    retry_counts = [0] * total_queries
    index_map = list(range(total_queries))

    remaining = queries[:]
    current_index_map = index_map[:]

    while remaining:
        batch_prompts, batch_images, batch_indices = [], [], []

        for i, q in enumerate(remaining):
            img = dataset[q['img_id']]['image']

            if q['question_type'] == 'ask_masked_obj':
                img = mask_object(img, q['sam'])
                img = img.convert("RGB")  

            elif q['question_type'] == 'ask_obj_color':
                img = box_object(turn_grayscale_image(img, '3channel'), q['sam'])
                img = img.convert("RGB")  
            
            batch_prompts.append(q['prompt'])
            batch_images.append(img)
            batch_indices.append(current_index_map[i])

            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1
                    raw_ans_list[idx].append(res)

                    pred = safe_extract_answer(res, queries[idx]['options'])
                    if pred:
                        acc_list[idx] = int(pred == queries[idx]['gt'].lower())
                        invalid_info_list[idx] = 0  # valid

                batch_prompts, batch_images, batch_indices = [], [], []

        # Retry logic: only continue for queries with no valid generation
        new_remaining, new_index_map = [], []
        for i in current_index_map:
            if invalid_info_list[i] == 1 and retry_counts[i] < queries[i]['max_retries']:
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    return raw_ans_list, acc_list, invalid_info_list


def group_per_question(all_queries, all_raw_ans, all_acc, all_invalid_info, dataset):
    """
    Group queries into per-question format:
    {
      "img_id": ...,
      "ground_truth_label": ...,
      "questions": [
        {
          "question_type": "ask_obj_color",
          "gt": "black",
          "acc": 0.75,
          "invalid_info": 1,
          "object_index": 0,
          "details": [
            {"prompt": ..., "raw_ans": [...], "acc": 1, "invalid_info": 0},
            ...
          ]
        },
        ...
      ]
    }
    """
    grouped = defaultdict(lambda: defaultdict(list))  # {img_id: {(type, obj_index, gt): [query_results]}}

    for q, raw_list, a, inv in zip(all_queries, all_raw_ans, all_acc, all_invalid_info):
        key = (q["img_id"], q["question_type"], q["object_index"], q["gt"])
        grouped[q["img_id"]][(q["question_type"], q["object_index"], q["gt"])].append({
            'prompt': q["prompt"],
            'raw_ans': raw_list,
            'acc': a,
            'invalid_info': inv,
            'options': q["options"],
        })

    result = []
    for img_id in grouped:
        img_entry = {
            "img_id": img_id,
            "ground_truth_label": dataset[img_id]['label'],
            "questions": []
        }

        for (qtype, obj_index, gt), query_list in grouped[img_id].items():
            avg_acc = sum(q['acc'] for q in query_list) / len(query_list)
            invalid_sum = sum(q['invalid_info'] for q in query_list)

            # Get confuser from first query’s options
            first_prompt = query_list[0]
            confuser = [opt for opt in first_prompt['options'] if opt != gt]

            # Drop options from each prompt detail to avoid redundancy
            for q in query_list:
                q.pop('options', None)

            img_entry["questions"].append({
                "question_type": qtype,
                "object_index": obj_index,
                "gt": gt,
                "confuser": confuser,
                "acc": avg_acc,
                "invalid_info": invalid_sum,
                "details": query_list
            })

        result.append(img_entry)

    return result


def main(args):
    # output all parameters
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")

    # save directory setting
    if args.ask_type == 'ordered_choice':
        save_dir = Path(f"ObjColor_exp/traverse_res/{args.filter_apply}/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}")
    elif args.ask_type == 'random_choice':
        save_dir = Path(f"ObjColor_exp/traverse_res/{args.filter_apply}/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # get dataset
    print("[INFO] Loading data...")
    dataset, dataset_length = get_data(args.data_name)

    # get model
    print("[INFO] Loading model...")
    model_dict = load_target_model(args)
    model_dict["image_cache"] = ImageCache(max_size=32)
    inference_func = INFERENCE_FUNCS[args.target_model] 

    # initialize logger and record all parameters
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    # save address
    save_path = save_dir / f'res.json'

    # timing
    start_time = time.time()

    # obtain queries 
    entries = read_json(args.confuser_json)
    all_queries = []
    for entry in tqdm(entries, desc='[Prepare Queries]'):
        all_queries.extend(prepare_queries(entry, args.ask_type, args.ask_time, args.ask_masked_obj_prompt_id, args.ask_obj_color_prompt_id))

    # process queries into chunks 
    print(f"[INFO] Running {len(all_queries)} queries in blocks with size {args.query_block_size}...")
    all_raw_ans, all_acc, all_invalid_info = [], [], []
    num_blocks = ceil(len(all_queries) / args.query_block_size)

    for i in tqdm(range(num_blocks), desc="[Blocks]"):
        start = i * args.query_block_size
        end = min((i+1) * args.query_block_size, len(all_queries))
        qblock = all_queries[start:end]
        block_raw_ans, block_acc, block_invalid_info = run_batch_query_with_retry(qblock, inference_func, model_dict, dataset, args)
        all_raw_ans.extend(block_raw_ans)
        all_acc.extend(block_acc)
        all_invalid_info.extend(block_invalid_info)

    # Aggregate by question
    grouped = group_per_question(all_queries, all_raw_ans, all_acc, all_invalid_info, dataset)
    save_json(save_path, grouped)

    # record
    time_notice = format_elapsed_time(start_time)
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")

    print("[INFO] Running AUC evaluation...")
    for qtype in ["ask_masked_obj", "ask_obj_color", "both"]:
        for ignore_flag in [False, True]:
            print(f"\n>>> AUC Eval | qtype={qtype} | ignore_invalid={ignore_flag}")
            logger.info(f"\n>>> AUC Eval | qtype={qtype} | ignore_invalid={ignore_flag}")

            if qtype == "both":
                # Merge both question types
                def merged_compute_auc(data, ignore_invalid):
                    scores, labels, label0_ids, label1_ids = [], [], [], []
                    for entry in data:
                        label = entry["ground_truth_label"]
                        img_id = entry["img_id"]
                        valid_qs = [
                            q for q in entry["questions"]
                            if (not ignore_invalid or q["invalid_info"] == 0)
                        ]
                        if not valid_qs:
                            continue
                        image_score = sum(q["acc"] for q in valid_qs) / len(valid_qs)
                        scores.append(image_score)
                        labels.append(label)
                        (label0_ids if label == 0 else label1_ids).append(img_id)
                    if len(set(labels)) < 2:
                        print("[WARN] Only one label class found. Skipping.")
                        return None, len(label0_ids), len(label1_ids)
                    
                    auc = roc_auc_score(labels, scores)
                    return auc, len(label0_ids), len(label1_ids)

                auc, n0, n1 = merged_compute_auc(grouped, ignore_invalid=ignore_flag)
            else:
                auc, n0, n1 = compute_auc(grouped, question_type=qtype, ignore_invalid=ignore_flag)

            if auc is not None:
                print(f"[AUC] {auc:.4f} | label0: {n0}, label1: {n1}")
                logger.info(f"[AUC] {auc:.4f} | label0: {n0}, label1: {n1}")
            else:
                print("[AUC] N/A")
                logger.info("[AUC] N/A")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--target_model', type=str, default='llava_v1_5_7b')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--ask_masked_obj_prompt_id', type=int, default=1)
    parser.add_argument('--ask_obj_color_prompt_id', type=int, default=1)
    parser.add_argument('--confuser_json', type=str, required=True, help='please provide the directory of confuser.json file')
    parser.add_argument('--filter_apply', type=str, choices=['withFilter', 'noFilter'], required=True, help='application of filter mechanism, must align with confuser_json') 
    parser.add_argument('--ask_type', type=str, default='ordered_choice', 
                                                choices=['ordered_choice', 'random_choice'],
                                                help=(
                                                    "Option order strategy: "
                                                    "'ordered_choice' cycles the correct answer through all 4 positions to remove bias; "
                                                    "'random_choice' randomly shuffles options each time (controlled by --ask_time)."
                                                    )
                                                )
    parser.add_argument('--ask_time', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--query_block_size', type=int, default=30)

    args = parser.parse_args()

    # Ensure ask_time is only required for random_choice
    if args.ask_type == 'random_choice' and args.ask_time is None:
        parser.error("--ask_time is required when --ask_type is 'random_choice'")
    if args.ask_type == 'ordered_choice':
        args.ask_time = None  # Set to None explicitly to avoid confusion

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)