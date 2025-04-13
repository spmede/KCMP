
import torch
from PIL import Image

from .path_utils import add_to_syspath
add_to_syspath("/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA/llama_adapter_v21")
import llama


# def llama_adapter_batch_inference(model_dict, input_queries, imgs, args):
#     """
#     Truly batched inference for LLaMA-Adapter.
    
#     Args:
#         model_dict: dict with 'model' and 'preprocess'
#         input_queries: list[str], prompts
#         imgs: list[PIL.Image] or torch.Tensor
#         args: config with .temperature, .top_p, .num_gen_token, .gpu_id
    
#     Returns:
#         list[str]: generated texts
#     """
#     model = model_dict['model']
#     preprocess = model_dict['preprocess']
#     device = f'cuda:{args.gpu_id}'

#     # 1. Preprocess all images into a batch
#     processed_imgs = []
#     for img in imgs:
#         if isinstance(img, str):
#             img = Image.open(img).convert("RGB")
#         if isinstance(img, Image.Image):
#             img_tensor = preprocess(img).unsqueeze(0)
#         elif isinstance(img, torch.Tensor):
#             img_tensor = img.unsqueeze(0) if img.ndim == 3 else img
#         else:
#             raise ValueError("Invalid image type.")
#         processed_imgs.append(img_tensor)
#     batch_imgs = torch.cat(processed_imgs, dim=0).to(device)

#     # 2. Format prompts
#     prompts = [llama.format_prompt(q) for q in input_queries]

#     # 3. Generate using model's batched `generate` method
#     outputs = model.generate(
#         imgs=batch_imgs,
#         prompts=prompts,
#         max_gen_len=args.num_gen_token,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         device=device,
#     )

#     return outputs


def llama_adapter_batch_inference(model_dict, input_queries, imgs, args):
    model = model_dict['model']
    preprocess = model_dict['preprocess']
    image_cache = model_dict.get("image_cache", None)
    device = f'cuda:{args.gpu_id}'

    prompts = [llama.format_prompt(q) for q in input_queries]

    processed_imgs = []
    for img in imgs:
        cached = image_cache.get(img) if image_cache else None
        if cached is not None:
            processed_imgs.append(cached)
        else:
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            if image_cache:
                image_cache.set(img, img_tensor)
            processed_imgs.append(img_tensor)

    batch_imgs = torch.cat(processed_imgs, dim=0)

    outputs = model.generate(
        imgs=batch_imgs,
        prompts=prompts,
        max_gen_len=args.num_gen_token,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )

    return outputs
