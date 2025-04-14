

'''问原图是否存在某颜色'''


import os
import re
import json
import time
import argparse
from PIL import Image
from math import ceil
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from eval_auc_generic import compute_auc
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
    '2': (
        "You are shown a grayscale version of an image. However, you have seen the original colored image during training.\n\n"
        "Which of the following colors was present in the original image?\n"
        "Choose exactly one option. Do NOT explain your choice. Respond only with the color.\n\n"
        "Options:\n{}\n{}\n{}\n{}\n\n"
        "Example format:\nAnswer: red\n\n"
        "Your Answer:"
    ),
    '3': (
        "You are given a grayscale image.\n"
        "You are asked to recall which color was present in the original version of the image that you have seen during training.\n\n"
        "Your task: From the list below, identify the ONE color that was present in the original image.\n"
        "Respond with only the color name. Do not explain.\n\n"
        "Options:\n{}\n{}\n{}\n{}\n\n"
        "Format your answer as:\nAnswer: [color]\n\n"
        "Example:\nAnswer: blue\n\n"
        "Now answer:\nAnswer:"
    ),
    '4': (
        "This image is shown in grayscale. But during training, you have seen its original colored version.\n\n"
        "Your task is to choose the correct color that appeared in the original image.\n"
        "Pick exactly ONE from the list. No explanation is needed.\n\n"
        "Color choices:\n{}\n{}\n{}\n{}\n\n"
        "Strict format: Answer: [color name]\n"
        "Example:\nAnswer: green\n\n"
        "Answer:"
    ),
    '5': (
        "You are given a grayscale image.\n"
        "You must recall which of the following colors was present in the original (colored) version that you saw during training.\n"
        "Please answer with ONLY the color name, nothing else.\n\n"
        "Available colors:\n{}\n{}\n{}\n{}\n\n"
        "Format:\nAnswer: <chosen color>\n"
        "Example:\nAnswer: purple\n\n"
        "Answer:"
    )
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

    # Step 3: fuzzy fallback - any full word match # 此处是否保留
    # for opt in options:
    #     if re.search(rf"\\b{re.escape(opt)}\\b", text):
    #         return opt

    return None

def prepare_queries(entry, ask_type, ask_time, prompt_id):
    queries = []
    for q_idx, item in enumerate(entry['color_choices']):
        gt = item['gt_color']
        distractors = item['confuser']
        options = [gt] + distractors
        if ask_type == 'ordered_choice':
            for i in range(4):
                reordered = options[:]
                reordered[0], reordered[i] = reordered[i], reordered[0]
                queries.append({
                    'prompt': build_prompt(reordered, prompt_id),
                    'img_id': entry['img_id'],
                    'question_index': q_idx,  # q_idx 标识该 prompt 所属的问题
                    'options': reordered,
                    'gt': gt,
                    'max_retries': 5
                })
        else:
            from random import shuffle
            for _ in range(ask_time):
                shuffled = options[:]
                shuffle(shuffled)
                queries.append({
                    'prompt': build_prompt(shuffled, prompt_id),
                    'img_id': entry['img_id'],
                    'question_index': q_idx,  # q_idx 标识该 prompt 所属的问题
                    'options': shuffled,
                    'gt': gt,
                    'max_retries': 1
                })
    return queries

def run_batch_query_with_retry(queries, inference_func, model_dict, dataset, args):
    total_queries = len(queries)
    outputs = []
    retry_counts = [0] * total_queries
    index_map = list(range(total_queries))

    remaining = queries[:]
    current_index_map = index_map[:]

    while remaining:
        batch_prompts, batch_images, batch_indices = [], [], []

        for i, q in enumerate(remaining):
            img = dataset[q['img_id']]['image']
            gray_img = turn_grayscale_image(img, manner='3channel')

            batch_prompts.append(q['prompt'])
            batch_images.append(gray_img)
            batch_indices.append(current_index_map[i])

            if len(batch_prompts) == args.batch_size or i == len(remaining) - 1:
                responses = inference_func(model_dict, batch_prompts, batch_images, args)

                for j, res in enumerate(responses):
                    idx = batch_indices[j]
                    retry_counts[idx] += 1
                    pred = safe_extract_answer(res, queries[idx]['options'])
                    if pred:
                        outputs.append({
                            'img_id': queries[idx]['img_id'],
                            'question_index': queries[idx]['question_index'],
                            'gt_color': queries[idx]['gt'],
                            'full_prompt': queries[idx]['prompt'],
                            'options': queries[idx]['options'],
                            'answer': pred,
                            'acc': int(pred == queries[idx]['gt'].lower()),     # 该条 query 的 acc
                            'invalid_info': 0                                   # 该条 query 的 invalid_info
                        })
                    elif retry_counts[idx] >= queries[idx]['max_retries']:
                        outputs.append({
                            'img_id': queries[idx]['img_id'],
                            'question_index': queries[idx]['question_index'],   # 每一条 query 根据 img_id, question_index 后续被 group 至所属 question 
                            'gt_color': queries[idx]['gt'],
                            'full_prompt': queries[idx]['prompt'],
                            'options': queries[idx]['options'],
                            'answer': 'NoAnswer',
                            'acc': 0,                                           # 该条 query 的 acc
                            'invalid_info': 1                                   # 该条 query 的 invalid_info
                        })

                batch_prompts, batch_images, batch_indices = [], [], []

        new_remaining, new_index_map = [], []
        for i in current_index_map:
            if retry_counts[i] < queries[i]['max_retries']:  # 这里判定对吗？我们应该只对 未超出 max_retries 次数且仍未生成valid答案的 query进行重新生成
                new_remaining.append(queries[i])
                new_index_map.append(i)

        remaining = new_remaining
        current_index_map = new_index_map

    return outputs


def group_per_question(outputs, dataset):
    grouped = defaultdict(lambda: defaultdict(list))
    for item in outputs:
        grouped[item['img_id']][item['question_index']].append(item)

    final_result = []
    for img_id, question_dict in grouped.items():
        temp = {'img_id': img_id,
                'ground_truth_label': dataset[img_id]['label'],
                'questions': []}
        for q_idx, qitems in question_dict.items():
            acc = sum(x['acc'] for x in qitems) / len(qitems)
            invalid = sum(x['invalid_info'] for x in qitems)
            temp['questions'].append({
                'question_index': q_idx,
                'gt_color': qitems[0]['gt_color'],
                'acc': acc, 
                'invalid_info': invalid,
                'details': qitems,  # 该条问题对应的所有 query 的回答情况
            })
        final_result.append(temp)
    return final_result


def main(args):
    # 输出所有参数
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")

    # 保存位置 + 记录运行时间
    if args.ask_type == 'ordered_choice':
        save_dir = Path(f"ImgColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    elif args.ask_type == 'random_choice':
        save_dir = Path(f"ImgColor_exp/res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}-{time.strftime('%Y%m%d-%H%M%S')}")
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

    entries = read_json(args.color_json)
    all_queries = []
    for entry in tqdm(entries, desc="[Prepare Queries]"):
        all_queries.extend(prepare_queries(entry, args.ask_type, args.ask_time, args.ask_imgcolor_prompt_id))

    # 分块处理 queries
    print(f"[INFO] Running {len(all_queries)} queries in blocks with size {args.query_block_size}...")
    all_outputs = []
    num_blocks = ceil(len(all_queries) / args.query_block_size)
    for i in tqdm(range(num_blocks), desc="[Blocks]"):
        start = i * args.query_block_size
        end = min((i+1) * args.query_block_size, len(all_queries))
        qblock = all_queries[start:end]
        block_outputs = run_batch_query_with_retry(qblock, inference_func, model_dict, dataset, args)
        all_outputs.extend(block_outputs)

    # Aggregate by question
    grouped = group_per_question(all_outputs, dataset)
    save_json(save_path, grouped)

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

        auc_info = compute_auc(grouped, ignore_flag=ignore_flag)

        if auc_info['auc'] is not None:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC={auc_info['auc']:.4f} | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC={auc_info['auc']:.4f} | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")
        else:
            print(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")
            logger.info(f"[AUC ignore_invalid={ignore_flag}] AUC=N/A | label0={len(auc_info['label0']['ids'])} label1={len(auc_info['label1']['ids'])}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--target_model', type=str, default='llava_v1_5_7b')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_gen_token', type=int, default=32)
    parser.add_argument('--ask_imgcolor_prompt_id', type=int, default=1)
    parser.add_argument('--color_json', type=str, required=True)  # confuser 存储位置
    parser.add_argument('--ask_type', type=str, default='ordered_choice', choices=['ordered_choice', 'random_choice'])
    parser.add_argument('--ask_time', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--query_block_size', type=int, default=1000)

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
    