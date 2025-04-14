

import os
import re
import json
import time
import argparse
from math import ceil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from eval_auc_0414 import compute_auc
from infer_utili.logging_func import *
from infer_utili.image_cache import ImageCache
from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json
from infer_utili.image_utils import turn_grayscale_image
from infer_utili import llava_batch_inference, minigpt4_batch_inference, llama_adapter_batch_inference


PROMPT_TEMPLATE = {
    '1': (
        "The image is in grayscale, but you have seen the original version during training.\n"
        "Which of the following colors appeared in the original image?\n\n"
        "Options:\n{}\n{}\n{}\n{}\n\n"
        "Answer:"
    ),
    }

INFERENCE_FUNCS = {
    'llava_v1_5_7b': llava_batch_inference,
    'MiniGPT4': minigpt4_batch_inference,
    'llama_adapter_v2': llama_adapter_batch_inference,
}

def build_prompt(options, prompt_id):
    return PROMPT_TEMPLATE[str(prompt_id)].format(*options)

def safe_extract_answer(text, options):
    text = text.strip().lower()
    options = [opt.lower() for opt in options]

    # Step 1: if one and only one option is mentioned, accept it
    matched = [opt for opt in options if opt in text]
    if len(matched) == 1:
        return matched[0]

    # Step 2: try extract answer from known patterns
    match = re.search(r"answer\s*[::\-]?\s*(\w+)", text, re.IGNORECASE)
    if match:
        response = match.group(1).strip().lower()
        for opt in options:
            if response == opt:
                return opt

    # Step 3: fuzzy fallback - any full word match
    for opt in options:
        if re.search(rf"\\b{re.escape(opt)}\\b", text):
            return opt

    return None

def prepare_queries(entry, ask_type, ask_time, prompt_id):
    queries = []
    for item in entry['color_choices']:
        gt = item['gt_color']
        distractors = item['confuser']
        options = [gt] + distractors
        if ask_type == 'ordered_choice':
            for i in range(4):
                reordered = options[:]
                reordered[0], reordered[i] = reordered[i], reordered[0]
                queries.append({
                    'img_id': entry['img_id'],
                    'question_type': 'ask_img_color',
                    'prompt': build_prompt(reordered, prompt_id),
                    'options': reordered,
                    'gt': gt,  # ImgColor 实验中可由 gt 确定所属的 question
                    'max_retries': 5
                })
        else:
            from random import shuffle
            for _ in range(ask_time):
                shuffled = options[:]
                shuffle(shuffled)
                queries.append({
                    'img_id': entry['img_id'],
                    'question_type': 'ask_img_color',
                    'prompt': build_prompt(shuffled, prompt_id),
                    'options': shuffled,
                    'gt': gt,
                    'max_retries': 1
                })
    return queries


def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    '''
    输入 queries 数量为 batch_size
    返回
    raw_ans: list[list[str]] 是list, 第 i 个元素是第 i 条 query 直到生成 valid_ans 以及 max_retries 次数内所有生成内容
    acc: list[int] 每条 query 的准确性, 元素取值仅为 0/1, 第 i 个元素是 第 i 条 query 的判断结果, 如果第 i 条 query 在 max_retries 次数内生成了 valid_ans, 根据 valid_ans 判断 acc=0/1; 如果达到 max_retries, acc=0
    invalid_info=[] 是list, 元素取值仅为 0/1, 第 i 个元素是 第 i 条 query 的判断结果, 如果第 i 条 query 超出 max_retries 仍无 valid ans, 值为 1, 否则为0
    '''
    total_queries = len(queries)
    raw_ans = [[] for _ in range(total_queries)]
    acc = [0] * total_queries
    invalid_info = [1] * total_queries  # 默认无效，若找到 valid ans 再置为 0
    retry_counts = [0] * total_queries
    index_map = list(range(total_queries))

    remaining = queries[:]
    current_index_map = index_map[:]

    while remaining:
        batch_prompts, batch_images, batch_indices = [], [], []

        for i, q in enumerate(remaining):
            image = dataset[q['img_id']]['image']
            gray_img = turn_grayscale_image(image, manner='3channel')

            batch_prompts.append(q['prompt'])
            batch_images.append(gray_img)
            batch_indices.append(current_index_map[i])

            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1
                    raw_ans[idx].append(res)

                    pred = safe_extract_answer(res, queries[idx]['options'])
                    if pred:  # valid answer
                        acc[idx] = int(pred == queries[idx]['gt'].lower())
                        invalid_info[idx] = 0  # 标记当前 query 有效，不再 retry

                batch_prompts, batch_images, batch_indices = [], [], []

        # 再筛选一次哪些 query 还需要 retry（未生成 valid 且还未超出 max_retries）
        new_remaining, new_index_map = [], []
        for i in current_index_map:
            if invalid_info[i] == 1 and retry_counts[i] < queries[i]['max_retries']:
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    return raw_ans, acc, invalid_info


def group_per_question(all_queries, all_raw_ans, all_acc, all_invalid_info, dataset, confuser_lookup):
    '''
    保存成下列格式
    result = [
    {'img_id': 0,
    'ground_truth_label': 1,
    
    'questions': [
        {'question_type': 'ask_img_color', 
        'acc': 0.3,  # mean value of all queries constrcued from this question
        'invalid_info': 2, # sum of invalid_info from all queries constrcued from this question 

        'gt': ,  # to locate which question, in ImgColor experiment, gt is enough to locate the question 
        'confuser': ,
        'details': # shows details about all queries constructed from this question
            [  
            {'prompt': , # full prompt for this query
            'raw_ans': , # raw answer for this query
            'acc': 1, # per query, only 0 or 1
            'invalid_info': 0, # per query, only 0 or 1
            }, {}, {}]
        }]
    },
    {'img_id': 1,
    'ground_truth_label': 0,
    'questions': ...
    }
    ]
    '''
    # Step 1: Group queries by (img_id, gt_color)
    grouped = defaultdict(lambda: defaultdict(list))  # {img_id: {gt_color: [entries]}}

    for q, raw_list, a, inv in zip(all_queries, all_raw_ans, all_acc, all_invalid_info):
        img_id = q['img_id']
        gt = q['gt']
        grouped[img_id][gt].append({
            'prompt': q['prompt'],
            'raw_ans': raw_list,
            'acc': a,
            'invalid_info': inv,
        })
    
    # Step 2: Construct final result list
    result = []
    for img_id in grouped:
        entry = {
            'img_id': img_id,
            'ground_truth_label': dataset[img_id]['label'],
            'questions': []
        }

        for gt_color, query_entries in grouped[img_id].items():
            acc_mean = sum(e['acc'] for e in query_entries) / len(query_entries)
            invalid_sum = sum(e['invalid_info'] for e in query_entries)

            confuser = confuser_lookup.get((img_id, gt_color), [])

            entry['questions'].append({
                'question_type': 'ask_img_color',
                'gt': gt_color,
                'confuser': confuser,
                'acc': acc_mean,
                'invalid_info': invalid_sum,
                'details': query_entries
            })

        result.append(entry)

    return result


def main(args):
    # 输出所有参数
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")

    # 保存位置 + 记录运行时间
    if args.ask_type == 'ordered_choice':
        save_dir = Path(f"ImgColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    elif args.ask_type == 'random_choice':
        save_dir = Path(f"ImgColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}-{time.strftime('%Y%m%d-%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # get dataset
    print("[INFO] Loading data...")
    dataset, dataset_length = get_data(args.data_name)

    # get model
    print("[INFO] Loading model...")
    model_dict = load_target_model(args)
    model_dict["image_cache"] = ImageCache(max_size=32)
    inference_func = INFERENCE_FUNCS[args.target_model] 

    # initialize logger 记录所有参数，包括target model 的部分设置 (e.g., llama adapter_dir)
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    # save address
    save_path = save_dir / f'res.json'

    # 计时
    start_time = time.time()

    # 获取所有 queries 
    entries = read_json(args.ImgColor_confuser_json)
    all_queries = []
    for entry in tqdm(entries, desc="[Prepare Queries]"):
        all_queries.extend(prepare_queries(entry, args.ask_type, args.ask_time, args.ask_imgcolor_prompt_id))


    # 分块处理 queries
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

    confuser_lookup = {
        (q['img_id'], q['gt']): [opt for opt in q['options'] if opt != q['gt']] 
        for q in all_queries
        }

    # Aggregate by question
    grouped = group_per_question(all_queries, all_raw_ans, all_acc, all_invalid_info, dataset, confuser_lookup)
    save_json(save_path, grouped)

    # 记录时间
    elapsed_time = time.time() - start_time
    h, m, s = int(elapsed_time // 3600), int(elapsed_time % 3600 // 60), int(elapsed_time % 60)
    time_notice = f"{h}h {m}min {s}sec" if h > 0 else f"{m}min {s}sec"
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")

    print("[INFO] Running AUC evaluation...")
    for ignore_flag in [False, True]:
        print(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")
        logger.info(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")

        auc, num_label0, num_label1 = compute_auc(grouped)

        if auc is not None:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC={auc:.4f} | label0={num_label0} label1={num_label1}")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC={auc:.4f} | label0={num_label0} label1={num_label1}")
        else:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--target_model', type=str, default='llava_v1_5_7b')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--ask_imgcolor_prompt_id', type=int, default=1)
    parser.add_argument('--ImgColor_confuser_json', type=str, required=True)  # confuser 存储位置
    parser.add_argument('--ask_type', type=str, default='ordered_choice', choices=['ordered_choice', 'random_choice'])
    parser.add_argument('--ask_time', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--query_block_size', type=int, default=100)

    # 测试用
    args = parser.parse_args()

    # Ensure ask_time is only required for random_choice
    if args.ask_type == 'random_choice' and args.ask_time is None:
        parser.error("--ask_time is required when --ask_type is 'random_choice'")
    if args.ask_type == 'ordered_choice':
        args.ask_time = None  # Set to None explicitly to avoid confusion

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
