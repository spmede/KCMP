

import re
import torch
import tempfile
import numpy as np
import pycocotools.mask as mask_util

from torchvision.ops import box_convert

from .path_utils import add_to_syspath
add_to_syspath("/data/Grounded-SAM-2")  # please change to the actual model location

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict, load_image



def clean_description(text):
    text = re.sub(r'[、，]', ',', text).lower().strip()
    
    valid_items = []
    for item in re.split(r'[,.]', text):
        item = item.strip()
        if not item:
            continue

        item = re.sub(r'[^a-z\. ]', '', item).strip()

        if not item.endswith('.'):
            item += '.'
        
        if 2 < len(item.replace('.', '')) < 25:
            valid_items.append(item)
    
    return ', '.join(list(set(valid_items)))


def get_SAM_model(gpu_id):

    GROUNDING_DINO_CONFIG = "/data/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "/data/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
    SAM2_CHECKPOINT = "/data/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=f"cuda:{gpu_id}"
    )

    # build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=f"cuda:{gpu_id}")
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    return grounding_model, sam2_model, sam2_predictor


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def SAM_inference(sam2_predictor, grounding_model, here_image, obj_name):

    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.4

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as temp_file:
        here_image.save(temp_file.name, format="JPEG")
        
        image_source, image = load_image(temp_file.name)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=obj_name,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
        )
    
    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False
        )
    
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    sam_result = [
        {
            "class_name": class_name,
            "bbox": box,
            "segmentation": mask_rle,
            "score": score,
        }
        for class_name, box, mask_rle, score in zip(labels, input_boxes, mask_rles, scores)
        ]

    return sam_result