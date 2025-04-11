

# chmod +x command_0.sh
# ./command_0.sh

# ['llava_v1_5_7b', 'MiniGPT4', 'llama_adapter_v2']
# ['img_Flickr', 'img_dalle']



CUDA_VISIBLE_DEVICES=0 python target_model_traverse.py \
    --data_name img_Flickr \
    --target_model llava_v1_5_7b \
    --gpu_id 0 \
    --temperature 0.3 \
    --top_p 0.9 \
    --num_gen_token 32 \
    --ask_obj_prompt_id 3 \
    --ask_color_prompt_id 3 \
    --confuser_json /data1/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/ObjColor_exp/confuser_res/img_Flickr/confuser_res.json \
    --ask_type random_choice \
    --ask_time 4 \
    --batch_size 4 \
    --query_block_size 10 \
    # --start_pos 0 \
    # --end_pos 3








