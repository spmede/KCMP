
import torch
from PIL import Image

from .path_utils import add_to_syspath
add_to_syspath("target_model/VL-MIA/llama_adapter_v21")
import llama


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


def llama_adapter_oneSample_inference(model_dict, input_query, img, args):
    '''single sample inference'''

    model = model_dict['model']
    preprocess = model_dict['preprocess']
    image_cache = model_dict.get("image_cache", None)
    device = f'cuda:{args.gpu_id}'

    cached = image_cache.get(img) if image_cache else None
    if cached is not None:
        img_tensor = cached
    else:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        if image_cache:
            image_cache.set(img, img_tensor)

    output = model.generate(
        imgs=img_tensor,
        prompts=[llama.format_prompt(input_query)],
        max_gen_len=args.num_gen_token,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )[0].strip()

    return [output]  # return list