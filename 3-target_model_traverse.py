

'''
2025.4.2
ask_obj + ask_color
实验步骤
1. SAM + Qwen 分析图像: Qwen 获取物体名称, SAM 获取物体位置信息

2. 混淆选项生成: 
1) mask 物体后由 Qwen 生成可能的物体名称; 
2) 框出物体后由 Qwen 生成物体颜色真值, 然后随机选择3个混淆颜色

3. target model inference: 模型回答问题, 两种提问方式. [此处实现第3步]
1) random_choice (提问n次, 每次提问时随机打乱4个选项的顺序); 
2) ordered_choice (正确答案依次放A/B/C/D进行询问)
'''

import os
import re
import json
import time
import torch
import random
import argparse

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

from img_utli import *
from llava_utili import *  # llava inference function
from logging_func import *

import sys
from pathlib import Path
paths_to_add = [
    Path("/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA/MiniGPT-4"),  # for miniGPT4
    Path("/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA/llama_adapter_v21"),  # for llama adapter
]
for custom_path in paths_to_add:
    if str(custom_path) not in sys.path:
        sys.path.append(str(custom_path))

import llama

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub



ask_obj_PROMPT = {
    '1': (
    "You have seen the image in your training data. Choose the option that correctly identifies the original content of the masked area. You must choose one option.\n"
    "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    ),
}

ask_color_PROMPT = {
    '1': (
    "The image is in grayscale, but you have encountered it during training. "
    "Identify the original color of the object enclosed in the red box. You must choose one option.\n"
    "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    ),
    # '2': (
    # "You have seen this image during your training phase. "
    # "In the original image, what was the color of the object inside the bounding box?\n"
    # "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    # ),
    # '3': (
    # "This is a grayscale image with an object highlighted by a red bounding box. "
    # "Assume you have encountered this image during the training phase. "
    # "What is the original color of the object within the box?\n"
    # "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    # ),
    # '4': (
    # "This is a grayscale version of an image you encountered during training. "
    # "The red box highlights an object. What was the original color of the object inside the box?\n"
    # "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    # ),
    '5': (
    "This is a grayscale version of an image you encountered during training. "
    "What is the original color of the object within the red box?\n"
    "Options:\n{}\n{}\n{}\n{}\nAnswer:"
    ),
}


def get_data(data_specific):
    used_dataset = load_dataset("/data/yinjinhua/LMdataset/VL-MIA-image", data_specific, split='train')
    # img_Flickr (600), img_Flickr_10k, img_Flickr_2k, img_dalle (592)
    dataset_length = len(used_dataset)
    # print(dataset_length)
    return used_dataset, dataset_length


def minigpt_inference(model_dict, input_query, img, args, do_sample=True):

    minigpt_model = model_dict['model']
    vis_processor = model_dict['processor']
    CONV_VISION = model_dict['CONV_VISION']

    chat = Interact(minigpt_model, vis_processor, device=f'cuda:{(args.gpu_id)}')
    img_list = []
    chat_state = CONV_VISION.copy()
    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)
    chat.ask(input_query, chat_state)

    gen_ = chat.get_generate_output(conv=chat_state,
                                    img_list=img_list,
                                    max_new_tokens=args.num_gen_token,
                                    do_sample=do_sample,
                                    temperature=args.temperature,
                                    top_p=args.top_p
                                    )
    output_text = chat.model.llama_tokenizer.decode(gen_[0], skip_special_tokens=True)

    return output_text


def llama_adapter_inference(model_dict, input_query, img, args):

    model = model_dict['model']
    preprocess = model_dict['preprocess']

    device = 'cuda:{}'.format(args.gpu_id)
    prompt = llama.format_prompt(input_query)
    img = preprocess(img).unsqueeze(0).to(device)

    output_text = model.generate(img, 
                                 [prompt], 
                                 max_gen_len=args.num_gen_token, 
                                 temperature=args.temperature, 
                                 top_p=args.top_p,
                                 device=device)[0]

    return output_text



def ordered_choice_query(used_img, true_ans, confuser_options, used_prompt, inference_func, model_dict, args):
    """Performs ordered-choice querying and returns results with accuracy."""

    all_options = [true_ans] + confuser_options
    orders = ['A', 'B', 'C', 'D']
    max_retries = 5
    res = []
    invalid_flag = 0

    for order in orders:
        order_idx = orders.index(order)
        shuffled_options = all_options[:]
        shuffled_options[0], shuffled_options[order_idx] = shuffled_options[order_idx], shuffled_options[0]

        retry = 0
        valid_judge = False
        while not valid_judge and retry < max_retries:
            prompt = used_prompt.format(*shuffled_options)

            # Dynamically call the correct inference function
            raw_answer = inference_func(model_dict, prompt, used_img, args)

            matched_options = [opt for opt in all_options if re.search(re.escape(str(opt)), raw_answer.lower())]
            if len(matched_options) == 1:
                res.append(matched_options[0])
                valid_judge = True
            else:
                retry += 1

        if retry == max_retries:
            invalid_flag += 1

    correct_count = sum(1 for ans in res if ans == true_ans)
    acc = correct_count / len(res) if res else 0
    return res, acc, invalid_flag


def ramdom_choice_query(used_img, true_ans, confuser_options, used_prompt, inference_func, model_dict, args):
    """Performs ramdom-choice querying and returns results with accuracy."""

    all_options = [true_ans] + confuser_options
    res = []
    invalid_flag = 0

    for _ in range(args.ask_time):
        random.shuffle(all_options)
        prompt = used_prompt.format(*all_options)

        # Dynamically call the correct inference function
        raw_answer = inference_func(model_dict, prompt, used_img, args)

        matched_options = [opt for opt in all_options if re.search(re.escape(str(opt)), raw_answer.lower())]
        if len(matched_options) == 1:
            res.append(matched_options[0])
        else:
            invalid_flag += 1
    
    correct_count = sum(1 for ans in res if ans == true_ans)
    acc = correct_count / len(res) if res else 0
    return res, acc, invalid_flag


def load_target_model(args):
    """Loads the target model based on args.target_model"""

    if args.target_model == 'llava-v1.5-7b':
        llava_model_add = '/data/yinjinhua/LMmodel/liuhaotian_llava-v1.5-7b'
        llava_model_base = None
        llava_model_name = get_model_name_from_path(llava_model_add)
        conv_mode = load_conversation_template(llava_model_name)
        llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_pretrained_model(
            llava_model_add, llava_model_base, llava_model_name, gpu_id=args.gpu_id)
        return llava_model, llava_tokenizer, llava_image_processor, conv_mode

    elif args.target_model == 'MiniGPT4':
        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
        args.cfg_path = "/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA/MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml"
        args.options = None
        cfg = Config(args)
        miniGPT_model_config = cfg.model_cfg
        miniGPT_model_config.device_8bit = args.gpu_id
        miniGPT_model_cls = registry.get_model_class(miniGPT_model_config.arch)
        miniGPT_model = miniGPT_model_cls.from_config(miniGPT_model_config).to(f'cuda:{args.gpu_id}')
        CONV_VISION = conv_dict[miniGPT_model_config.model_type]  # miniGPT_model_config.model_type = pretrain_llama2
        miniGPT_vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        miniGPT_vis_processor = registry.get_processor_class(miniGPT_vis_processor_cfg.name).from_config(miniGPT_vis_processor_cfg)
        return miniGPT_model, miniGPT_vis_processor, CONV_VISION

    elif args.target_model == 'llama_adapter_v2':
        llama_dir = '/data/yinjinhua/LMmodel/yangchen_llama2-7B'
        adapter_dir = [
            '/data/yinjinhua/NLP/5-VLLM_MIA/target_model/model_weight/LORA-BIAS-7B-v21.pth',  # 默认用这个
            '/data/yinjinhua/NLP/5-VLLM_MIA/target_model/model_weight/LORA-BIAS-7B.pth',
            '/data/yinjinhua/NLP/5-VLLM_MIA/target_model/model_weight/CAPTION-7B.pth',
            '/data/yinjinhua/NLP/5-VLLM_MIA/target_model/model_weight/BIAS-7B.pth',
        ][0] 
        args.adapter_dir = adapter_dir
        llama_adapter_model, llama_adapter_preprocess = llama.load(
            adapter_dir, llama_dir, llama_type="7B", device=f'cuda:{args.gpu_id}')
        llama_adapter_model.eval()
        return llama_adapter_model, llama_adapter_preprocess

    else:
        raise ValueError(f"Unknown target model: {args.target_model}")



def main(args):

    # 输出所有参数
    print(f"******* {', '.join(f'{k}: {v}' for k, v in vars(args).items())} *******")

    # 保存位置一并记录运行时间
    if args.ask_type == 'ordered_choice':
        path = Path(f"res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}-{time.strftime('%Y%m%d-%H%M%S')}")
    elif args.ask_type == 'random_choice':
        path = Path(f"res/{args.data_name}/{args.target_model}/{args.ask_type}/temp_{args.temperature}_askTime_{args.ask_time}-{time.strftime('%Y%m%d-%H%M%S')}")
    path.mkdir(parents=True, exist_ok=True)


    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)


    # Load the target model
    model_components = load_target_model(args)

    # Assign model-specific variables
    if args.target_model == 'llava-v1.5-7b':
        llava_model, llava_tokenizer, llava_image_processor, conv_mode = model_components
        model_dict = {
            "model": llava_model,
            "tokenizer": llava_tokenizer,
            "image_processor": llava_image_processor,
            "conv_mode": conv_mode
            }
        
    elif args.target_model == 'MiniGPT4':
        miniGPT_model, miniGPT_vis_processor, CONV_VISION = model_components
        model_dict = {"model": miniGPT_model, "processor": miniGPT_vis_processor, 'CONV_VISION': CONV_VISION}

    elif args.target_model == 'llama_adapter_v2':
        llama_adapter_model, llama_adapter_preprocess = model_components
        model_dict = {"model": llama_adapter_model, "preprocess": llama_adapter_preprocess}

    # Define the inference function mapping
    inference_function = {
        'llava-v1.5-7b': llava_inference,
        'MiniGPT4': minigpt_inference,
        'llama_adapter_v2': llama_adapter_inference
        }[args.target_model]


    # initialize logger 记录所有参数，包括target model 的部分设置 (e.g., llama adapter_dir)
    log_filename = os.path.join(path, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=%s", args.__dict__) 

    # save address
    # start_pos = 0 
    # end_pos = dataset_length
    start_pos = args.start_pos
    end_pos = args.end_pos
    save_add = os.path.join(path, f'res_start_{start_pos}_end_{end_pos}.json')


    # 读取 confuser result
    confuser_add = f'/data/yinjinhua/NLP/5-VLLM_MIA/11-obj_color/confuser_res/{args.data_name}/confuser_res.json'
    with open(confuser_add, 'r') as f:
        confuser_res = json.load(f)  # keys: ['original_img_id', 'ground_truth_label', 'object_name', 'sam_result']

    # target model inference
    inference_result = []

    for sample_id in tqdm(range(start_pos, end_pos)):

        # 输出保存为 json, 其中图片用 id 指示
        current_img_id, current_img_label, current_image = sample_id, used_dataset[sample_id]['label'], used_dataset[sample_id]['image']
        current_sam_list = confuser_res[sample_id]['sam_result'] 
        # list, 含该张图片所有检测到的物体信息及 confuser 信息
        # element keys: ['class_name', 'bbox', 'segmentation', 'score', 'obj_name_distractors', 'obj_gt_color', 'obj_gt_color_distractors']
        
        current_img_dict = confuser_res[sample_id].copy()

        # 该张图片有检测到物体
        if current_sam_list is not None and len(current_sam_list) > 0:
            # Create a new list to store the updated sam results
            updated_sam_list = []

            for single_sam_res in current_sam_list: 

                # 1. ask object, 对图片mask处的物体进行提问
                obj_name_distractors = single_sam_res['obj_name_distractors']
                # 对该物体成功生成了3个混淆选项
                if obj_name_distractors is not None:

                    # 获取 mask img 
                    masked_image = mask_object(current_image, single_sam_res)

                    if args.ask_type == 'ordered_choice':
                        ask_obj_model_ans, ask_obj_model_acc, ask_obj_invalid_info = ordered_choice_query(
                            used_img=masked_image, 
                            true_ans=single_sam_res['class_name'], 
                            confuser_options=obj_name_distractors, 
                            used_prompt=ask_obj_PROMPT[args.ask_obj_prompt_id], 
                            inference_func=inference_function, 
                            model_dict=model_dict, 
                            args=args, 
                        )

                    elif args.ask_type == 'random_choice':
                        ask_obj_model_ans, ask_obj_model_acc, ask_obj_invalid_info = ramdom_choice_query(
                            used_img=masked_image, 
                            true_ans=single_sam_res['class_name'], 
                            confuser_options=obj_name_distractors, 
                            used_prompt=ask_obj_PROMPT[args.ask_obj_prompt_id], 
                            inference_func=inference_function, 
                            model_dict=model_dict, 
                            args=args, 
                        )

                    # Add ask obj res to the single_sam_res
                    single_sam_res['ask_obj_ans'] = ask_obj_model_ans
                    single_sam_res['ask_obj_acc'] = ask_obj_model_acc
                    single_sam_res['ask_obj_invalid_info'] = ask_obj_invalid_info


                # 2. ask color, 对灰度图 框中物体的颜色进行提问
                obj_gt_color = single_sam_res['obj_gt_color']
                # 对该物体成功提取出了颜色
                if obj_gt_color is not None:

                    obj_gt_color_distractors = single_sam_res['obj_gt_color_distractors']

                    # 图片转灰度图
                    if args.target_model == 'MiniGPT4':
                        grayscale_image = turn_grayscale_image(current_image, manner='3channel')
                    else:
                        grayscale_image = turn_grayscale_image(current_image, manner='default')
                    # 框标记物体
                    grayscale_image_with_box = box_object(grayscale_image, single_sam_res, box_color='red', box_width=3)

                    if args.ask_type == 'ordered_choice':
                        ask_color_model_ans, ask_color_model_acc, ask_color_invalid_info = ordered_choice_query(
                            used_img=grayscale_image_with_box, 
                            true_ans=obj_gt_color, 
                            confuser_options=obj_gt_color_distractors, 
                            used_prompt=ask_color_PROMPT[args.ask_color_prompt_id], 
                            inference_func=inference_function, 
                            model_dict=model_dict, 
                            args=args, 
                        )
                    
                    elif args.ask_type == 'random_choice':
                        ask_color_model_ans, ask_color_model_acc, ask_color_invalid_info = ramdom_choice_query(
                            used_img=grayscale_image_with_box, 
                            true_ans=obj_gt_color, 
                            confuser_options=obj_gt_color_distractors, 
                            used_prompt=ask_color_PROMPT[args.ask_color_prompt_id], 
                            inference_func=inference_function, 
                            model_dict=model_dict, 
                            args=args, 
                        )

                    # Add ask obj res to the single_sam_res
                    single_sam_res['ask_color_ans'] = ask_color_model_ans
                    single_sam_res['ask_color_acc'] = ask_color_model_acc
                    single_sam_res['ask_color_invalid_info'] = ask_color_invalid_info


                # Add the updated single_sam_res to the updated list
                updated_sam_list.append(single_sam_res)
            
            # Update the sam_result in current_img_dict with the updated list
            current_img_dict['sam_result'] = updated_sam_list

        inference_result.append(current_img_dict)

        # save confuser generation per image
        with open(save_add, 'w') as f:
            json.dump(inference_result, f, indent=4)



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr', 
                        choices=['img_Flickr', 'img_Flickr_10k', 'img_Flickr_2k', 
                                 'img_dalle'])
    
    parser.add_argument('--target_model', default='llava-v1.5-7b', 
                        choices=['llava-v1.5-7b', 'MiniGPT4', 'llama_adapter_v2'])
    
    parser.add_argument('--gpu_id', default=1, type=int, help='visible gpu ids')

    parser.add_argument('--temperature', default=0.3, type=float, help='generation param')
    parser.add_argument('--top_p', default=0.9, type=float, help='generation param')
    parser.add_argument('--num_gen_token', default=32, type=int, help='generation param')

    parser.add_argument("--ask_obj_prompt_id", type=str, default=1, help='used prompt for ask obj')
    parser.add_argument("--ask_color_prompt_id", type=str, default=1, help='used prompt for ask color')

    parser.add_argument("--ask_type", type=str, default='ordered_choice', 
                        choices=['ordered_choice', 'random_choice'])

    parser.add_argument('--ask_time', type=int, default=None, 
                        help='repeat ask number per question (only used for random_choice)')

    parser.add_argument('--start_pos', type=int, help='start position of dataset')
    parser.add_argument('--end_pos', type=int, help='end position of dataset')

    args = parser.parse_args()

    # Ensure ask_time is only required for random_choice
    if args.ask_type == 'random_choice' and args.ask_time is None:
        parser.error("--ask_time is required when --ask_type is 'random_choice'")
    if args.ask_type == 'ordered_choice':
        args.ask_time = None  # Set to None explicitly to avoid confusion

    return args


if __name__ == "__main__":
    args = args_parse()
    main(args)


# python 3-target_model_traverse.py --data_name img_Flickr --target_model llava-v1.5-7b --gpu_id 1 --temperature 0.3 --top_p 0.9 --ask_type ordered_choice --start_pos 0 --end_pos 1

# python 3-target_model_traverse.py --data_name img_Flickr --target_model llava-v1.5-7b --gpu_id 1 --temperature 0.3 --top_p 0.9 --ask_type random_choice --ask_time 10 --start_pos 0 --end_pos 1

# python 3-target_model_traverse.py \
#     --data_name img_Flickr \
#     --target_model llava-v1.5-7b \
#     --gpu_id 1 \
#     --temperature 0.3 \
#     --top_p 0.9 \
#     --num_gen_token 32 \
#     --ask_obj_prompt_id 1 \
#     --ask_color_prompt_id 1\
#     --ask_type ordered_choice \
#     --start_pos 0 \
#     --end_pos 1


# python 3-target_model_traverse.py \
#     --data_name img_Flickr \
#     --target_model llava-v1.5-7b \
#     --gpu_id 1 \
#     --temperature 0.3 \
#     --top_p 0.9 \
#     --num_gen_token 32 \
#     --ask_obj_prompt_id 1 \
#     --ask_color_prompt_id 1\
#     --ask_type random_choice \
#     --ask_time 10 \
#     --start_pos 0 \
#     --end_pos 1


