import os
import re
import json
import time
import argparse
from tqdm import tqdm
from math import ceil
from pathlib import Path
from collections import defaultdict

from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json
from infer_utili.image_cache import ImageCache
from infer_utili.image_utils import reorder_image_patches
from infer_utili import llava_batch_inference, minigpt4_batch_inference, llama_adapter_batch_inference
from infer_utili.logging_func import init_logger

INFERENCE_FUNCS = {
    'llava_v1_5_7b': llava_batch_inference,
    'MiniGPT4': minigpt4_batch_inference,
    'llama_adapter_v2': llama_adapter_batch_inference,
}

def safe_extract_answer(text):
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    text = text.strip().upper()
    if text in label_map:
        return label_map[text]
    match = re.search(r'Answer[:\s]*([A-D])', text)
    if match:
        return label_map.get(match.group(1))
    return None

def prepare_all_queries(entry, ask_type, ask_time):
    from random import shuffle
    queries = []
    for q_idx, q in enumerate(entry['questions']):
        gt = q['gt_caption']
        confusers = q['confusers']
        all_options = [gt] + confusers
        # reorder_idx = q.get('reorder_index', q_idx)
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
                    # 'reorder_index': reorder_idx,
                    'gt_index': gt_index,
                    'prompt': prompt,
                    'max_retries': 5
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
                    # 'reorder_index': reorder_idx,
                    'gt_index': gt_index,
                    'prompt': prompt,
                    'max_retries': 1
                })
    return queries

def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    total_queries = len(queries)
    final_outputs = [[] for _ in range(total_queries)]
    valid_flags = [False] * total_queries
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
            # image = turn_grayscale_image(image, '3channel').convert("RGB")  # 要改
            image = reorder_image_patches(image, n=3, perm=q['perm'])
            batch_prompts.append(q['prompt'])
            batch_images.append(image)
            batch_indices.append(current_index_map[i])

            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1
                    final_outputs[idx].append(res)

                    if res == "[GENERATION_FAILED]":
                        failed_query_count += 1
                        failed_img_ids.add(queries[idx]['img_id'])
                        continue

                    pred_index = safe_extract_answer(res)  # 提取的是 A/B/C/D index
                    if pred_index is not None and 0 <= pred_index < 4:
                        valid_flags[idx] = True

                batch_prompts, batch_images, batch_indices = [], [], []

        new_remaining, new_index_map = [], []
        for i, valid in zip(current_index_map, valid_flags):
            if not valid and retry_counts[i] < queries[i]['max_retries']:
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    return final_outputs, valid_flags, retry_counts, {
        'failed_queries': failed_query_count,
        'failed_img_ids': list(failed_img_ids)
    }

def aggregate_results(queries, outputs, valids):
    result_map = defaultdict(list)
    for q, res_list, valid in zip(queries, outputs, valids):
        correct = sum(safe_extract_answer(res) == q['gt_index'] for res in res_list)
        acc = correct / len(res_list) if len(res_list) > 0 else 0
        result_map[q['img_id']].append({
            'reorder_index': q['reorder_index'],
            'perm': q['perm'],
            'prompt': q['prompt'],
            'gt_index': q['gt_index'],
            'answers': res_list,
            'acc': acc,
            'invalid_info': int(not valid)
        })
    return result_map

def main(args):
    save_dir = Path(f"CaptionConfuse_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/prompt_{args.desc_prompt_id}_temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset, _ = get_data(args.data_name)
    caption_data = read_json(args.caption_confuser_json)

    model_dict = load_target_model(args)
    model_dict["image_cache"] = ImageCache(max_size=32)
    inference_func = INFERENCE_FUNCS[args.target_model]

    logger = init_logger(os.path.join(save_dir, 'log.txt'), logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    queries = []
    for entry in caption_data:
        queries.extend(prepare_all_queries(entry, args.ask_type, args.ask_time))

    print(f"[INFO] Running {len(queries)} queries in blocks...")
    result_map = {}
    total_failed_queries = 0
    failed_img_ids = set()
    num_blocks = ceil(len(queries) / args.query_block_size)

    for i in tqdm(range(num_blocks), desc="[Running blocks]"):
        start = i * args.query_block_size
        end = min((i + 1) * args.query_block_size, len(queries))
        qblock = queries[start:end]
        outputs, valids, retries, stats = run_batch_query_with_retry(qblock, inference_func, model_dict, dataset, args)
        batch_result = aggregate_results(qblock, outputs, valids)
        for k, v in batch_result.items():
            result_map.setdefault(k, []).extend(v)
        total_failed_queries += stats['failed_queries']
        failed_img_ids.update(stats['failed_img_ids'])

    result_data = []
    for entry in caption_data:
        img_id = entry['img_id']
        result_data.append({
            'img_id': img_id,
            'ground_truth_label': entry['ground_truth_label'],
            'questions': result_map.get(img_id, [])
        })

    save_path = save_dir / f"res.json"
    save_json(save_path, result_data)
    print(f"[INFO] Results saved to: {save_path}")
    print(f"[SUMMARY] Total failed queries: {total_failed_queries}")
    print(f"[SUMMARY] Unique images failed: {len(failed_img_ids)}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--caption_confuser_json', type=str, required=True)
    parser.add_argument('--target_model', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--desc_prompt_id', type=int, required=True)
    parser.add_argument('--ask_type', type=str, choices=['ordered_choice', 'random_choice'], default='ordered_choice')
    parser.add_argument('--ask_time', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--query_block_size', type=int, default=1000)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
