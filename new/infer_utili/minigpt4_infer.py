
import torch
from PIL import Image

# from .image_utils import safe_prompts
from .path_utils import add_to_syspath
add_to_syspath("/data/yinjinhua/NLP/5-VLLM_MIA/target_model/VL-MIA/MiniGPT-4")
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2


# def minigpt4_batch_inference(model_dict, input_queries, imgs, args):
#     """
#     Truly batched MiniGPT-4 inference using internal generate().

#     Args:
#         model_dict: dict containing "model" (MiniGPT4), etc.
#         input_queries: List[str], each prompt contains <ImageHere> as placeholder
#         imgs: List[PIL.Image] or List[Tensor], same length as input_queries
#         args: object with temperature, top_p, num_gen_token, gpu_id
#     Returns:
#         List[str]: generated texts
#     """

#     model = model_dict["model"]
#     processor = model_dict["processor"]
#     device = f"cuda:{args.gpu_id}"

#     input_queries = safe_prompts(input_queries)

#     # Step 1: preprocess all images -> convert to tensor
#     processed_imgs = []
#     for img in imgs:
#         if isinstance(img, str):
#             img = Image.open(img).convert("RGB")
#         if isinstance(img, Image.Image):
#             img_tensor = processor(img).unsqueeze(0).to(device)  # [1, C, H, W]
#         elif isinstance(img, torch.Tensor):
#             img_tensor = img.unsqueeze(0).to(device) if len(img.shape) == 3 else img.to(device)
#         else:
#             raise TypeError("Unsupported image input.")
#         processed_imgs.append(img_tensor)

#     images_tensor = torch.cat(processed_imgs, dim=0)  # [B, C, H, W]

#     # Step 2: call MiniGPT4.generate()
#     generated_texts = model.generate(
#         images=images_tensor,
#         texts=input_queries,
#         temperature=args.temperature,
#         top_p=args.top_p,
#         max_new_tokens=args.num_gen_token,
#         do_sample=True
#     )

#     return generated_texts


# def minigpt4_batch_inference(model_dict, input_queries, imgs, args):
    # model = model_dict["model"]
    # processor = model_dict["processor"]
    # image_cache = model_dict.get("image_cache", None)
    # device = f"cuda:{args.gpu_id}"

    # from infer_utili.image_utils import safe_prompts
    # input_queries = safe_prompts(input_queries)

    # processed_imgs = []
    # for img in imgs:
    #     cached = image_cache.get(img) if image_cache else None
    #     if cached is not None:
    #         processed_imgs.append(cached)
    #     else:
    #         img_tensor = processor(img).unsqueeze(0).to(device)
    #         if image_cache:
    #             image_cache.set(img, img_tensor)
    #         processed_imgs.append(img_tensor)

    # images_tensor = torch.cat(processed_imgs, dim=0)

    # generated_texts = model.generate(
    #     images=images_tensor,
    #     texts=input_queries,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    #     max_new_tokens=args.num_gen_token,
    #     do_sample=True
    # )

    # return generated_texts


# 定位哪张图片推理时出错
def minigpt4_batch_inference(model_dict, input_queries, imgs, args):
    """
    Truly batched MiniGPT-4 inference with error tracking.
    Catches and prints failing prompts/images for debugging.
    """

    model = model_dict["model"]
    processor = model_dict["processor"]
    image_cache = model_dict.get("image_cache", None)
    device = f"cuda:{args.gpu_id}"

    from infer_utili.image_utils import safe_prompts
    input_queries = safe_prompts(input_queries)

    processed_imgs = []
    for img in imgs:
        cached = image_cache.get(img) if image_cache else None
        if cached is not None:
            processed_imgs.append(cached)
        else:
            try:
                img_tensor = processor(img).unsqueeze(0).to(device)
                if image_cache:
                    image_cache.set(img, img_tensor)
                processed_imgs.append(img_tensor)
            except Exception as e:
                print(f"[ERROR] Failed to process image: {img}")
                raise e

    images_tensor = torch.cat(processed_imgs, dim=0)

    try:
        with torch.no_grad():
            generated_texts = model.generate(
                images=images_tensor,
                texts=input_queries,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.num_gen_token,
                do_sample=True
            )
            return generated_texts

    except Exception as e:
        print("\n[❌ ERROR] MiniGPT-4 generation failed!")
        for i, (prompt, img) in enumerate(zip(input_queries, imgs)):
            print(f"\n--- Problematic Query [{i}] ---")
            print(f"Prompt: {prompt}")
            print(f"Image Mode: {img.mode}, Size: {img.size}, Format: {img.format}")
            try:
                extrema = img.getextrema()
                print(f"Image extrema: {extrema}")
            except Exception:
                print("Unable to get image extrema.")
        raise e

