
# 4.13 检查，有地方需要调整

import os
import re
import json
import time
import argparse
from tqdm import tqdm
from math import ceil
from pathlib import Path
from collections import defaultdict

from infer_utili.logging_func import *
from infer_utili.image_cache import ImageCache
from eval_auc_generic import compute_auc
from infer_utili.image_utils import reorder_image_patches
from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json
from infer_utili import llava_batch_inference, minigpt4_batch_inference, llama_adapter_batch_inference


INFERENCE_FUNCS = {
    'llava_v1_5_7b': llava_batch_inference,
    'MiniGPT4': minigpt4_batch_inference,
    'llama_adapter_v2': llama_adapter_batch_inference,
}

def safe_extract_answer(text):
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    text = text.strip().upper()

    # First, find all mentions of A/B/C/D (e.g., "A", "Answer: B", etc.)
    matches = re.findall(r'\b[A-D]\b', text)

    # If exactly one unique choice is found → return it
    unique_matches = list(set(matches))
    if len(unique_matches) == 1:
        return label_map[unique_matches[0]]

    # Otherwise, ambiguous or invalid
    return None


def prepare_all_queries(entry, ask_type, ask_time):
    from random import shuffle
    queries = []
    for q_idx, q in enumerate(entry['questions']):
        gt = q['gt_caption']
        confusers = q['confusers']
        all_options = [gt] + confusers
        perm = q.get('perm', [])

        if ask_type == 'ordered_choice':
            for i in range(4):
                reordered = all_options[:]
                reordered[0], reordered[i] = reordered[i], reordered[0]
                gt_index = i
                labels = ['A', 'B', 'C', 'D']
                prompt = (
                    "You have seen the original image. Which of the following descriptions best matches it?\n\n" +
                    "\n".join([f"{labels[j]}. {desc}" for j, desc in enumerate(reordered)]) +
                    "\n\nAnswer (A/B/C/D):"
                )
                queries.append({
                    'img_id': entry['img_id'],
                    'perm': perm,
                    'gt_index': gt_index,
                    'prompt': prompt,
                    'max_retries': 5,
                    'question_index': q_idx
                })
        else:
            for _ in range(ask_time):
                randomized = all_options[:]
                shuffle(randomized)
                gt_index = randomized.index(gt)
                labels = ['A', 'B', 'C', 'D']
                prompt = (
                    "You have seen the original image. Which of the following descriptions best matches it?\n\n" +
                    "\n".join([f"{labels[j]}. {desc}" for j, desc in enumerate(randomized)]) +
                    "\n\nAnswer (A/B/C/D):"
                )
                queries.append({
                    'img_id': entry['img_id'],
                    'perm': perm,
                    'gt_index': gt_index,
                    'prompt': prompt,
                    'max_retries': 1,
                    'question_index': q_idx
                })
    return queries

def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    total_queries = len(queries)
    final_outputs = ["NoAnswer"] * total_queries
    invalid_flags = [1] * total_queries
    retry_counts = [0] * total_queries
    index_map = list(range(total_queries))

    failed_query_count = 0
    failed_img_ids = set()

    remaining = queries[:]
    current_index_map = index_map[:]

    while remaining:
        batch_prompts, batch_images, batch_indices = [], [], []

        for i, q in enumerate(remaining):
            image = dataset[q['img_id']]['image']
            image, _ = reorder_image_patches(image, n=3, perm=q['perm'])
            batch_prompts.append(q['prompt'])
            batch_images.append(image)
            batch_indices.append(current_index_map[i])

            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1

                    if res == "[GENERATION_FAILED]":
                        failed_query_count += 1
                        failed_img_ids.add(queries[idx]['img_id'])
                        continue

                    pred_index = safe_extract_answer(res)
                    if pred_index is not None and 0 <= pred_index < 4:
                        final_outputs[idx] = res
                        invalid_flags[idx] = 0

                batch_prompts, batch_images, batch_indices = [], [], []

        new_remaining, new_index_map = [], []
        for i in current_index_map:
            if invalid_flags[i] == 1 and retry_counts[i] < queries[i]['max_retries']:
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    return final_outputs, invalid_flags, retry_counts, {
        'failed_queries': failed_query_count,
        'failed_img_ids': list(failed_img_ids)
    }

def aggregate_results(queries, outputs, invalids):
    result_map = defaultdict(lambda: defaultdict(list))

    for q, res, inv in zip(queries, outputs, invalids):
        pred_index = safe_extract_answer(res)
        acc = int(pred_index == q['gt_index']) if pred_index is not None else 0
        result_map[q['img_id']][q['question_index']].append({
            'prompt': q['prompt'],
            'gt_index': q['gt_index'],
            'perm': q['perm'],
            'answer': res,
            'acc': acc,
            'invalid_info': inv
        })

    final_map = defaultdict(list)
    for img_id in result_map:
        for q_idx in result_map[img_id]:
            prompts = result_map[img_id][q_idx]
            acc = sum(p['acc'] for p in prompts) / len(prompts)  # 已经归一化
            invalid_sum = sum(p['invalid_info'] for p in prompts)
            final_map[img_id].append({
                'perm': prompts[0]['perm'],
                'acc': acc,
                'invalid_info': invalid_sum,
                'raw': prompts
            })
    return final_map

def main(args):
    # 输出所有参数
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")

    # 保存位置 + 记录运行时间
    if args.ask_type == 'ordered_choice':
        save_dir = Path(f"Patch_exp/res/{args.data_name}/{args.target_model}/prompt_{args.desc_prompt_id}/{args.ask_type}/temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    elif args.ask_type == 'random_choice':
        save_dir = Path(f"Patch_exp/res/{args.data_name}/{args.target_model}/prompt_{args.desc_prompt_id}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}-{time.strftime('%Y%m%d-%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Loading data...")
    dataset, dataset_length = get_data(args.data_name)

    print("[INFO] Loading model...")
    model_dict = load_target_model(args)
    model_dict["image_cache"] = ImageCache(max_size=32)
    inference_func = INFERENCE_FUNCS[args.target_model]

    # initialize logger 记录所有参数，包括target model 的部分设置 (e.g., llama adapter_dir)
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    # save address
    start_pos = 0 
    end_pos = dataset_length
    save_path = save_dir / f'res_start_{start_pos}_end_{end_pos}.json'

    # 计时
    start_time = time.time()

    caption_data = read_json(args.caption_confuser_json)
    queries = []
    for entry in tqdm(caption_data, desc="[Prepare Queries]"):
        queries.extend(prepare_all_queries(entry, args.ask_type, args.ask_time))

    print(f"[INFO] Running {len(queries)} queries in blocks...")
    result_map = {}
    num_blocks = ceil(len(queries) / args.query_block_size)
    total_failed_queries = 0
    failed_img_ids = set()

    for i in tqdm(range(num_blocks), desc="[Blocks Progress]"):
        start = i * args.query_block_size
        end = min((i + 1) * args.query_block_size, len(queries))
        qblock = queries[start:end]
        outputs, invalids, retries, stats = run_batch_query_with_retry(qblock, inference_func, model_dict, dataset, args)
        batch_result = aggregate_results(qblock, outputs, invalids)
        for k, v in batch_result.items():
            result_map.setdefault(k, []).extend(v)

        total_failed_queries += stats['failed_queries']
        failed_img_ids.update(stats['failed_img_ids'])

    print(f"[SUMMARY] Total failed queries: {total_failed_queries}")
    print(f"[SUMMARY] Unique images affected: {len(failed_img_ids)} / {dataset_length}")
    logger.info(f"[SUMMARY] Total failed queries: {total_failed_queries}")
    logger.info(f"[SUMMARY] Unique images affected: {len(failed_img_ids)} / {dataset_length}")
    logger.info(f"[SUMMARY] Unique images affected index: {failed_img_ids}")

    result_data = []
    for entry in caption_data:
        img_id = entry['img_id']
        result_data.append({
            'img_id': img_id,
            'ground_truth_label': entry['ground_truth_label'],
            'questions': result_map.get(img_id, [])
        })

    save_json(save_path, result_data)

    # 记录时间
    elapsed_time = time.time() - start_time
    h, m, s = int(elapsed_time // 3600), int(elapsed_time % 3600 // 60), int(elapsed_time % 60)
    time_notice = f"{h}h {m}min {s}sec" if h > 0 else f"{m}min {s}sec"
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")

    # === Compute AUC automatically ===
    print("[INFO] Running AUC evaluation...")
    for ignore_flag in [False, True]:
        print(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")
        logger.info(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")

        auc_info = compute_auc(result_data, ignore_flag=ignore_flag)

        if auc_info['auc'] is not None:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC={auc_info['auc']:.4f} | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC={auc_info['auc']:.4f} | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")
        else:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--target_model', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--desc_prompt_id', type=int, required=True)
    parser.add_argument('--caption_confuser_json', type=str, required=True)
    parser.add_argument('--ask_type', type=str, choices=['ordered_choice', 'random_choice'], default='ordered_choice')
    parser.add_argument('--ask_time', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--query_block_size', type=int, default=1000)
    args = parser.parse_args()

    # Ensure ask_time is only required for random_choice
    if args.ask_type == 'random_choice' and args.ask_time is None:
        parser.error("--ask_time is required when --ask_type is 'random_choice'")
    if args.ask_type == 'ordered_choice':
        args.ask_time = None  # Set to None explicitly to avoid confusion

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)








