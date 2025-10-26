
import os
import time
import json
import argparse

from tqdm import tqdm
from pathlib import Path

from infer_utili.sam_utli import *
from infer_utili.logging_func import *
from infer_utili.apiCallFunc import apiCall_img
from infer_utili.data_utili import get_data, save_json, format_elapsed_time

# prompt for image analysis 
PROMPT_obj_analysis = (
        'You are a professional image analyst. Describe images strictly following these rules:\n'
        '1. List only clearly visible main objects\n'
        '2. Use English singular noun form\n'
        '3. Separate objects with commas, each ending with a period\n'
        '4. Order by visual significance\n'
        '5. No adjectives, colors or locations\n\n'
        'Example: person., dog., car., tree., fire hydrant.\n'
        'Answer:'
        )

def main(args):

    # get dataset
    used_dataset, dataset_length = get_data(args.data_name)

    # save add
    save_dir = Path(f"ObjColor_exp/img_analysis_res/{args.data_name}_by_{args.vllm_APImodel}")
    save_dir.mkdir(parents=True, exist_ok=True)

    start_pos, end_pos = 0, dataset_length
    save_path = save_dir / f'res.json'

    # initialize logger and record all parameters
    log_filename = os.path.join(save_dir, 'log.txt')
    logger = init_logger(log_filename, logging.INFO)
    logger.info("args=\n%s", json.dumps(args.__dict__, indent=4))

    # get SAM model
    grounding_model, sam2_model, sam2_predictor = get_SAM_model(args.gpu_id)

    # timing
    start_time = time.time()

    result = []
    total_fail_index = []

    for sample_id in tqdm(range(start_pos, end_pos)):

        # save res in json format
        current_img_id, current_img_label, current_image = sample_id, used_dataset['label'][sample_id], used_dataset[sample_id]['image']
        
        max_retry = 5
        retry_count = 0
        sam_result = None

        while retry_count < max_retry:
            try:
                # Step 1: analyse image and get object name
                object_name = apiCall_img(current_image, 
                                          PROMPT_obj_analysis, 
                                          temperature=0.6,
                                          num_gen_token=64, 
                                          model=args.vllm_APImodel)
                object_name_refined = clean_description(object_name)

                # is no objects are detected (object_name_refined is empty), sam_result=None
                if not object_name_refined:
                    sam_result = None
                else:
                    # Step 2: SAM analysis
                    sam_result = SAM_inference(sam2_predictor=sam2_predictor,
                                               grounding_model=grounding_model,
                                               here_image=current_image, 
                                               obj_name=object_name_refined)

                break

            except Exception as e:
                retry_count += 1
        
        # max_retry exceeded
        if sam_result is None:
            total_fail_index.append(sample_id)

        # save above infomation
        current_img_dict = dict(original_img_id=current_img_id, 
                                ground_truth_label=current_img_label,
                                object_name=object_name_refined,
                                sam_result=sam_result)
        
        # save after every image analysis
        result.append(current_img_dict)
        save_json(save_path, result)

    # record time
    elapsed_time = format_elapsed_time(start_time)
    print(f"[INFO] Inference complete. Consuming {elapsed_time}. Results saved to: {save_path}")
    logger.info(f"[INFO] Inference complete. Consuming {elapsed_time}. Results saved to: {save_path}")

    print(f"[INFO] Failed index number {len(total_fail_index)}. Failed indices: {total_fail_index}")
    logger.info(f"[INFO] Failed index number {len(total_fail_index)}. Failed indices: {total_fail_index}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='img_Flickr', 
                        choices=['img_Flickr', 'img_Flickr_10k', 'img_Flickr_2k', 'img_dalle'])
    
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--vllm_APImodel', type=str, default='gpt-4o-mini')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)