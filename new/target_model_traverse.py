import os
import re
import json
import time
import argparse
from math import ceil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from eval_auc import eval_from_json
from infer_utili.logging_func import *
from infer_utili.image_cache import ImageCache
from infer_utili.load_model_utili import load_target_model
from infer_utili.data_utili import get_data, read_json, save_json
from infer_utili.image_utils import mask_object, box_object, turn_grayscale_image
from infer_utili import llava_batch_inference, minigpt4_batch_inference, llama_adapter_batch_inference


PROMPT_TEMPLATES = {
    'ask_obj': {
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
    'ask_color': {
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

def build_prompt(template, options):
    return template.format(*options)

def extract_valid_answer(text, options):
    matched = [opt for opt in options if re.search(rf"\b{re.escape(opt)}\b", text.lower())]
    return matched[0] if len(matched) == 1 else None


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


def prepare_all_queries(img_id, sam, prompt_type, prompt_template, ask_type, ask_time):
    from random import shuffle
    gt = sam['class_name'] if prompt_type == 'ask_obj' else sam['obj_gt_color']
    distractors = sam['obj_name_distractors'] if prompt_type == 'ask_obj' else sam['obj_gt_color_distractors']
    options = [gt] + distractors

    queries = []
    if ask_type == 'ordered_choice':
        for i in range(4):
            reordered = options[:]
            reordered[0], reordered[i] = reordered[i], reordered[0]
            prompt = build_prompt(prompt_template, reordered)
            queries.append({
                'prompt': prompt,
                'img_id': img_id,
                'options': [opt.lower() for opt in reordered],
                'gt': gt.lower(),
                'max_retries': 5,
                'task_type': prompt_type
            })
    else:  # random_choice
        for _ in range(ask_time):
            randomized = options[:]
            shuffle(randomized)
            prompt = build_prompt(prompt_template, randomized)
            queries.append({
                'prompt': prompt,
                'img_id': img_id,
                'options': [opt.lower() for opt in randomized],
                'gt': gt.lower(),
                'max_retries': 1,
                'task_type': prompt_type
            })
    return queries

def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    """
    对所有 queries 批量推理 (含 retry)
    进度条按 query 完成更新 (无论是否 valid)
    """
    total_queries = len(queries)
    final_outputs = [[] for _ in range(total_queries)]
    valid_flags = [False for _ in range(total_queries)]
    retry_counts = [0 for _ in range(total_queries)]
    index_map = list(range(total_queries))
    failed_query_count = 0
    failed_img_ids = set()

    # pbar = tqdm(total=total_queries, desc="[Inference Progress]")  # 一批内是否加进度条
    remaining = queries[:]
    current_index_map = index_map[:]

    while remaining:
        batch_prompts, batch_images, batch_indices = [], [], []

        batch_options = [] # 自己加的

        for i, q in enumerate(remaining):
            image = dataset[q['img_id']]['image']
            sam = q['sam']
            if q['task_type'] == 'ask_obj':
                image = mask_object(image, sam)
            else:
                image = box_object(turn_grayscale_image(image, '3channel'), sam)
            
            image = image.convert("RGB")

            batch_prompts.append(q['prompt'])
            batch_images.append(image)
            batch_indices.append(current_index_map[i])

            batch_options.append(q['options']) # 自己加的

            # 满一批就推理
            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)
                # print(responses)  # 打印 raw ans
                # print(batch_options) # 打印 选项

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1
                    final_outputs[idx].append(res)

                    if res == "[GENERATION_FAILED]":  # 该 query 触发 miniGPT4 错误
                        failed_query_count += 1
                        failed_img_ids.add(queries[idx]['img_id'])
                        continue

                    # extracted = extract_valid_answer(res, remaining[idx]['options'])  # only strict match
                    extracted = safe_extract_answer(res, queries[idx]['options'])  # 扩充抽取答案的策略 + 改为全局索引引用
                    if extracted:
                        valid_flags[idx] = True
                    # else:
                        # print(f"[WARN] Not matched: res={res}, options={queries[idx]['options']}")

                # 无论成功失败，都算作完成一轮尝试，进度条更新
                # pbar.update(len(batch_prompts))

                batch_prompts, batch_images, batch_indices = [], [], []

        # 构造下一轮 retry
        new_remaining, new_index_map = [], []
        for i, valid in zip(current_index_map, valid_flags):
            if not valid and retry_counts[i] < queries[i]['max_retries']:
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    # pbar.close()

    # print(f"\n[INFO] Inference finished. Valid: {sum(valid_flags)} / {total_queries}")  # 汇报 valid 情况
    # return final_outputs, valid_flags, retry_counts
    return final_outputs, valid_flags, retry_counts, {
    'failed_queries': failed_query_count,
    'failed_img_ids': list(failed_img_ids)}

def aggregate_results(queries, outputs, valids, retries):
    """
    将所有 query 的推理结果聚合成每个物体的问题结果 (含准确率/invalid)
    """
    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'invalid': 0, 'answers': []})

    for q, outs, valid, retry in zip(queries, outputs, valids, retries):
        key = (q['img_id'], q['sam_id'], q['task_type'])
        correct = sum(1 for out in outs if q['gt'] in out.lower())
        stats[key]['answers'].extend(outs)
        stats[key]['total'] += len(outs)
        stats[key]['correct'] += correct
        if not valid:
            stats[key]['invalid'] += 1

    result_dict = {}
    for (img_id, sam_id, task_type), stat in stats.items():
        acc = stat['correct'] / (stat['total'] - stat['invalid']) if (stat['total'] - stat['invalid']) > 0 else 0
        result_dict[(img_id, sam_id, task_type)] = {
            f'{task_type}_ans': stat['answers'],
            f'{task_type}_acc': acc,
            f'{task_type}_invalid_info': stat['invalid']
        }
    return result_dict

def main(args):
    # 输出所有参数
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")
    
    # 保存位置 + 记录运行时间
    if args.ask_type == 'ordered_choice':
        save_dir = Path(f"ObjColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    elif args.ask_type == 'random_choice':
        save_dir = Path(f"ObjColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}-{time.strftime('%Y%m%d-%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading data...")
    dataset, dataset_length = get_data(args.data_name)
    confuser_res = read_json(args.confuser_json)

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
    # start_pos = args.start_pos
    # end_pos = args.end_pos
    save_path = save_dir / f'res_start_{start_pos}_end_{end_pos}.json'

    # 计时
    start_time = time.time()

    queries = []
    print("[INFO] Preparing queries...")
    for i in tqdm(range(start_pos, end_pos), desc="[Building Queries]"):
    # for i in tqdm(range(len(dataset)), desc="[Building Queries]"):
        img_entry = confuser_res[i]
        for j, sam in enumerate(img_entry.get('sam_result') or []):
            for task_type, prompt_id_key in [('ask_obj', 'ask_obj_prompt_id'), ('ask_color', 'ask_color_prompt_id')]:
                if task_type == 'ask_obj' and not sam.get('obj_name_distractors'):
                    continue
                if task_type == 'ask_color' and (not sam.get('obj_gt_color') or not sam.get('obj_gt_color_distractors')):
                    continue
                prompt_template = PROMPT_TEMPLATES[task_type][str(getattr(args, prompt_id_key))]
                q = prepare_all_queries(i, sam, task_type, prompt_template, args.ask_type, args.ask_time)
                for query in q:
                    query['sam'] = sam
                    query['sam_id'] = j
                queries.extend(q)

    # 一次传入所有 queries 数量太大会严重影响速度
    # print(f"[INFO] Running {len(queries)} queries in batch...")
    # outputs, valids, retries = run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args)
    # result_map = aggregate_results(queries, outputs, valids, retries)

    # 分块处理 queries
    print(f"[INFO] Running {len(queries)} queries in blocks of {args.query_block_size}...")
    result_map = {}
    total_queries = len(queries)
    num_blocks = ceil(total_queries / args.query_block_size)

    total_failed_queries = 0
    all_failed_img_ids = set()

    for block_idx in tqdm(range(num_blocks), desc="[Blocks Progress]"):
        q_start = block_idx * args.query_block_size
        q_end = min((block_idx + 1) * args.query_block_size, total_queries)
        # print(f"\n[INFO] Running block {block_idx + 1}/{num_blocks} | Queries {q_start} ~ {q_end}")

        query_block = queries[q_start:q_end]
        # outputs, valids, retries = run_batch_query_with_retry(query_block, inference_func, model_dict, dataset, args)
        outputs, valids, retries, block_stat = run_batch_query_with_retry(query_block, inference_func, model_dict, dataset, args)
        total_failed_queries += block_stat['failed_queries']
        all_failed_img_ids.update(block_stat['failed_img_ids'])
        result_map.update(aggregate_results(query_block, outputs, valids, retries))

    print(f"[SUMMARY] Total failed queries: {total_failed_queries}")
    print(f"[SUMMARY] Unique images affected: {len(all_failed_img_ids)} / {dataset_length}")
    logger.info(f"[SUMMARY] Total failed queries: {total_failed_queries}")
    logger.info(f"[SUMMARY] Unique images affected: {len(all_failed_img_ids)} / {dataset_length}")
    logger.info(f"[SUMMARY] Unique images affected index: {all_failed_img_ids}")

    final_result = []
    for i, img_entry in enumerate(confuser_res):
        for j, sam in enumerate(img_entry.get('sam_result') or []):
            for task_type in ['ask_obj', 'ask_color']:
                key = (i, j, task_type)
                if key in result_map:
                    sam.update(result_map[key])
        final_result.append(img_entry)
        save_json(save_path, final_result)

    
    # 记录时间
    elapsed_time = time.time() - start_time
    h, m, s = int(elapsed_time // 3600), int(elapsed_time % 3600 // 60), int(elapsed_time % 60)
    time_notice = f"{h}h {m}min {s}sec" if h > 0 else f"{m}min {s}sec"
    print(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {time_notice}. Results saved to: {save_path}")

    # evaluate AUC
    print("\n[INFO] Running AUC Evaluation...")
    for ignore_flag in [False, True]:
        print(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")
        logger.info(f"\n>>> AUC Summary (ignore_invalid={ignore_flag})")
        auc_result = eval_from_json(str(save_path), ignore_invalid=ignore_flag)
        for mode, v in auc_result.items():
            if v['auc'] is not None:
                print(f"  {mode:<10} | AUC: {v['auc']:.4f} | label0: {v['label0_num']} | label1: {v['label1_num']}")
                logger.info(f"  {mode:<10} | AUC: {v['auc']:.4f} | label0: {v['label0_num']} | label1: {v['label1_num']}")
            else:
                print(f"  {mode:<10} | AUC: N/A       | label0: {v['label0_num']} | label1: {v['label1_num']}")
                logger.info(f"  {mode:<10} | AUC: N/A       | label0: {v['label0_num']} | label1: {v['label1_num']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--target_model', default='llava_v1_5_7b')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--ask_obj_prompt_id', type=int, default=1)
    parser.add_argument('--ask_color_prompt_id', type=int, default=1)
    parser.add_argument('--confuser_json', type=str, required=True)  # confuser 存储位置
    parser.add_argument('--ask_type', type=str, default='ordered_choice', choices=['ordered_choice', 'random_choice'])
    parser.add_argument('--ask_time', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--query_block_size', type=int, default=1000, help='how many queries to process per retry block')


    # 测试用
    parser.add_argument('--start_pos', type=int, default=None, help='start position of dataset')
    parser.add_argument('--end_pos', type=int, default=None, help='end position of dataset')
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
