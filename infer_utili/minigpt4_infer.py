
import torch
from PIL import Image

from .path_utils import add_to_syspath
add_to_syspath("target_model/VL-MIA/MiniGPT-4")
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.interact import Interact, CONV_VISION_Vicuna0, CONV_VISION_LLama2



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
    

def minigpt4_oneSample_inference(model_dict, input_query, img, args):
    '''single sample inference'''

    model = model_dict["model"]
    processor = model_dict["processor"]
    CONV_VISION = model_dict["CONV_VISION"]
    image_cache = model_dict.get("image_cache", None)
    
    chat = Interact(model, processor, device='cuda:{}'.format(args.gpu_id))
    img_list = []
    chat_state = CONV_VISION.copy()
    llm_message = chat.upload_img(img, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask(input_query, chat_state)

    gen_ = chat.get_generate_output(conv=chat_state,
                                img_list=img_list,
                                do_sample=True,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                max_new_tokens=args.num_gen_token,
                                )

    output_text = chat.model.llama_tokenizer.decode(gen_[0], skip_special_tokens=True).strip()

    return [output_text]   # return list