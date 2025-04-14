

# chmod +x command_0.sh
# ./command_0.sh

# ['llava_v1_5_7b', 'MiniGPT4', 'llama_adapter_v2']
# ['img_Flickr', 'img_dalle']
# ['ordered_choice', 'random_choice']



# Patch exp

CUDA_VISIBLE_DEVICES=0 python target_model_caption_confuse_traverse.py \
    --data_name img_Flickr \
    --target_model llava_v1_5_7b \
    --gpu_id 0 \
    --temperature 0.3 \
    --top_p 0.9 \
    --num_gen_token 32 \
    --desc_prompt_id 1 \
    --caption_confuser_json /data/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/ObjColor_exp/confuser_res/img_Flickr/confuser_res.json \
    --ask_type random_choice \
    --ask_time 8 \
    --batch_size 10 \
    --query_block_size 30









# ObjColor exp
# CUDA_VISIBLE_DEVICES=3 python target_model_traverse.py \
#     --data_name img_Flickr \
#     --target_model llava_v1_5_7b \
#     --gpu_id 0 \
#     --temperature 0.3 \
#     --top_p 0.9 \
#     --num_gen_token 32 \
#     --ask_obj_prompt_id 3 \
#     --ask_color_prompt_id 3 \
#     --confuser_json /data/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/ObjColor_exp/confuser_res/img_Flickr/confuser_res.json \
#     --ask_type random_choice \
#     --ask_time 4 \
#     --batch_size 4 \
#     --query_block_size 10 \
#     --start_pos 0 \
#     --end_pos 3




# ImgColor exp
# for prompt_id in 2 3 4 5; do
#     CUDA_VISIBLE_DEVICES=3 python target_model_color_exist.py \
#         --data_name img_Flickr \
#         --target_model llava_v1_5_7b \
#         --gpu_id 0 \
#         --temperature 0.3 \
#         --top_p 0.9 \
#         --num_gen_token 32 \
#         --ask_imgcolor_prompt_id $prompt_id \
#         --color_json /data/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/ImgColor_exp/confuser_res/img_Flickr/confuser_res.json \
#         --ask_type random_choice \
#         --ask_time 8 \
#         --batch_size 2 \
#         --query_block_size 10
# done




# for temp in 0.3 0.5 0.7; do
#     for ask_obj_prompt_id in 1 2; do
#         for ask_color_prompt_id in 1 2; do
#             CUDA_VISIBLE_DEVICES=7 python target_model_traverse.py \
#                 --data_name img_dalle \
#                 --target_model llama_adapter_v2 \
#                 --gpu_id 0 \
#                 --temperature $temp \
#                 --top_p 0.9 \
#                 --num_gen_token 32 \
#                 --ask_obj_prompt_id $ask_obj_prompt_id \
#                 --ask_color_prompt_id $ask_color_prompt_id \
#                 --confuser_json /public/bupt_data/wangwd/yinjinhua/NLP/5-VLLM_MIA/12-batch_codebase/ObjColor_exp/confuser_res/img_dalle/confuser_res.json \
#                 --ask_type random_choice \
#                 --ask_time 6 \
#                 --batch_size 16 \
#                 --query_block_size 50 
#         done
#     done
# done


