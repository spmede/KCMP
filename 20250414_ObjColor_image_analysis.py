

import os
import torch
import argparse

from tqdm import tqdm
from infer_utili.data_utili import get_data, save_json
from infer_utili.sam_utli import *

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# Qwen inference function
def qwen_inference(qwen_model, qwen_processor, img, prompt, temperature, top_p, num_gen_token):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                },
                {"type": "text", 
                 "text": prompt},
            ],
        }
    ]

    # Preparation for inference
    text = qwen_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(qwen_model.device)

    # Inference: Generation of the output
    generated_ids = qwen_model.generate(**inputs, 
                                        max_new_tokens=num_gen_token,
                                        temperature=temperature,
                                        top_p=top_p)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

# prompt for image analysis 
PROMPT_obj_analysis = (
        'You are a professional image analyst. Describe images strictly following these rules:\n'
        '1. List only clearly visible main objects\n'
        '2. Use English singular noun forms\n'
        '3. Separate objects with commas, each ending with a period\n'
        '4. Order by visual significance\n'
        '5. No adjectives, colors or locations\n\n'
        'Example: person., dog., car., tree., fire hydrant.\n'
        'Answer:'
        )


def main(args):
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # get Qwen model
    qwen_model_add = "/data/yinjinhua/LMmodel/Qwen_Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(qwen_model_add, torch_dtype="auto", device_map=DEVICE)
    qwen_processor = AutoProcessor.from_pretrained(qwen_model_add)

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # save add
    save_add = f'ObjColor_exp/img_analysis_res/{args.data_name}/analysis_res.json'
    os.makedirs(os.path.dirname(save_add), exist_ok=True)

    # get SAM model
    grounding_model, sam2_model, sam2_predictor = get_SAM_model(args.gpu_id)


    result = []
    total_fail_index = []
    for sample_id in tqdm(range(dataset_length)):
        # 输出保存为 json, 其中图片用 id 指示
        current_img_id, current_img_label, current_image = sample_id, used_dataset['label'][sample_id], used_dataset[sample_id]['image']
        
        max_retry = 10
        retry_count = 0
        sam_result = None
        while retry_count < max_retry:
            try:
                # Step 1: analyse image and get object name
                object_name = qwen_inference(
                    qwen_model=qwen_model, 
                    qwen_processor=qwen_processor, 
                    img=current_image, 
                    prompt=PROMPT_obj_analysis, 
                    temperature=0.6, 
                    top_p=0.9, 
                    num_gen_token=64)
                
                object_name_refined = clean_description(object_name)

                # 如果未检测到物体 (object_name_refined 是空字符串), sam_result=None
                if not object_name_refined:
                    sam_result = None
                else:
                    # Step 2: SAM 分析
                    sam_result = SAM_inference(
                        sam2_predictor=sam2_predictor,
                        grounding_model=grounding_model,
                        here_image=current_image, 
                        obj_name=object_name_refined
                        )

                # 无异常时退出循环
                break

            except Exception as e:
                retry_count += 1
        
        # 超出 max_retry 仍失败的 sample
        if sam_result is None:
            total_fail_index.append(sample_id)

        # save above infomation
        current_img_dict = dict(original_img_id=current_img_id, 
                                ground_truth_label=current_img_label,
                                object_name=object_name_refined,
                                sam_result=sam_result,
                                )
        
        # save after every image analysis
        result.append(current_img_dict)

        # save result
        save_json(save_add, result)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


