
import time
import json
from datasets import load_dataset


def get_data(data_specific):
    used_dataset = load_dataset("VL-MIA-image", data_specific, split='train')
    # img_Flickr (600), img_Flickr_10k, img_Flickr_2k, img_dalle (592)
    dataset_length = len(used_dataset)
    # print(dataset_length)
    return used_dataset, dataset_length

def read_json(json_add):
    with open(json_add, 'r') as f:
        result = json.load(f)
    return result

def save_json(json_add, save_result):
    with open(json_add, 'w') as f:
        json.dump(save_result, f, indent=4)

def format_elapsed_time(start_time):
    """Return a formatted string of elapsed time since start_time."""
    elapsed = time.time() - start_time
    h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
    if h > 0:
        return f"{h}h {m}min {s}sec"
    else:
        return f"{m}min {s}sec"
    

