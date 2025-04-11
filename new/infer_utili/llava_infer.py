

import torch
from .image_utils import load_images
from .path_utils import add_to_syspath
add_to_syspath("/data1/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA")
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
# from llava.mm_utils import process_images, tokenizer_image_token
# from llava.conversation import conv_templates


def load_conversation_template(model_name):
    if "llama-2" in model_name.lower():
        return "llava_llama_2"
    elif "mistral" in model_name.lower():
        return "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        return "chatml_direct"
    elif "v1" in model_name.lower():
        return "llava_v1"
    elif "mpt" in model_name.lower():
        return "mpt"
    else:
        return "llava_v0"
    

# def llava_batch_inference(model_dict, input_queries, imgs, args, do_sample=True):
#     """
#     input_queries: list of strings
#     imgs: list of PIL images (or paths)
#     """

#     llava_model = model_dict["model"] 
#     llava_tokenizer = model_dict["tokenizer"]
#     image_processor = model_dict["image_processor"] 
#     conv_mode = model_dict["conv_mode"] 

#     image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

#     prompts = []
#     for qs in input_queries:
#         if IMAGE_PLACEHOLDER in qs:
#             if llava_model.config.mm_use_im_start_end:
#                 qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
#             else:
#                 qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
#         else:
#             if llava_model.config.mm_use_im_start_end:
#                 qs = image_token_se + "\n" + qs
#             else:
#                 qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#         conv = conv_templates[conv_mode].copy()
#         conv.append_message(conv.roles[0], qs)
#         conv.append_message(conv.roles[1], None)
#         prompts.append(conv.get_prompt())

#     # process images
#     images = load_images(imgs)  # list of PIL images
#     image_sizes = [x.size for x in images]
#     images_tensor = process_images(
#         images,
#         image_processor,
#         llava_model.config
#     ).to(llava_model.device, dtype=torch.float16)  # [B, C, H, W]

#     # tokenize prompts
#     input_ids_list = []
#     for prompt in prompts:
#         input_ids, _ = tokenizer_image_token(prompt, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         input_ids_list.append(input_ids)

#     # pad input_ids to same length
#     input_ids_padded = torch.nn.utils.rnn.pad_sequence(
#         input_ids_list, batch_first=True, padding_value=llava_tokenizer.pad_token_id
#     ).to(f'cuda:{args.gpu_id}')

#     # generate
#     with torch.inference_mode():
#         output_ids = llava_model.generate(
#             input_ids_padded,
#             images=images_tensor,
#             image_sizes=image_sizes,
#             do_sample=do_sample,
#             temperature=args.temperature, 
#             top_p=args.top_p, 
#             max_new_tokens=args.num_gen_token,
#         )

#     output_texts = llava_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     output_texts = [text.strip() for text in output_texts]
#     return output_texts




def llava_batch_inference(model_dict, input_queries, imgs, args, do_sample=True):
    llava_model = model_dict["model"] 
    tokenizer = model_dict["tokenizer"]
    processor = model_dict["image_processor"] 
    conv_mode = model_dict["conv_mode"]
    image_cache = model_dict.get("image_cache", None)

    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
    from llava.conversation import conv_templates
    import re

    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    prompts = []
    for qs in input_queries:
        if IMAGE_PLACEHOLDER in qs:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = image_token_se + "\n" + qs
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    # encode images with cache
    encoded_images = []
    image_sizes = []
    for img in imgs:
        cached = image_cache.get(img) if image_cache else None
        if cached:
            images_tensor, size = cached
        else:
            images_tensor = process_images([img], processor, llava_model.config).to(llava_model.device, dtype=torch.float16)
            size = img.size
            if image_cache:
                image_cache.set(img, (images_tensor, size))
        encoded_images.append(images_tensor)
        image_sizes.append(size)

    images_tensor = torch.cat(encoded_images, dim=0)

    # prompts -> input_ids
    input_ids_list = []
    for prompt in prompts:
        input_ids, _ = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids_list.append(input_ids)

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    ).to(f'cuda:{args.gpu_id}')

    output_ids = llava_model.generate(
        input_ids_padded,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=do_sample,
        temperature=args.temperature, 
        top_p=args.top_p, 
        max_new_tokens=args.num_gen_token,
    )

    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [t.strip() for t in output_texts]

