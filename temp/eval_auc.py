import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

def extract_per_image_scores(result_data, mode, ignore_invalid=False):
    image_scores = []
    image_labels = []
    image_ids = []
    label0_info = {'ids': [], 'scores': []}
    label1_info = {'ids': [], 'scores': []}

    for img_id, entry in enumerate(result_data):
        label = entry['ground_truth_label']
        sam_list = entry.get('sam_result') or []
        acc_list = []

        for sam in sam_list:
            # -- ask_obj
            if mode == 'ask_obj':
                if 'ask_obj_acc' not in sam:
                    continue
                if ignore_invalid and sam.get('ask_obj_invalid_info', 0) > 0:
                    continue
                acc_list.append(sam['ask_obj_acc'])

            # -- ask_color
            elif mode == 'ask_color':
                if 'ask_color_acc' not in sam:
                    continue
                if ignore_invalid and sam.get('ask_color_invalid_info', 0) > 0:
                    continue
                acc_list.append(sam['ask_color_acc'])

            # -- both
            elif mode == 'both': # 两个任务谁存在就统计谁的 acc, ignore_invalid=True 时排除掉 invalid 的那一部分
                temp_acc = []

                if 'ask_obj_acc' in sam:
                    if ignore_invalid and sam.get('ask_obj_invalid_info', 0) > 0:
                        pass  # skip
                    else:
                        temp_acc.append(sam['ask_obj_acc'])

                if 'ask_color_acc' in sam:
                    if ignore_invalid and sam.get('ask_color_invalid_info', 0) > 0:
                        pass
                    else:
                        temp_acc.append(sam['ask_color_acc'])

                if temp_acc:
                    acc_list.append(np.mean(temp_acc))


        if acc_list:
            mean_score = np.mean(acc_list)
            image_scores.append(mean_score)
            image_labels.append(label)
            image_ids.append(img_id)

            if label == 0:
                label0_info['ids'].append(img_id)
                label0_info['scores'].append(mean_score)
            else:
                label1_info['ids'].append(img_id)
                label1_info['scores'].append(mean_score)

    return np.array(image_scores), np.array(image_labels), image_ids, label0_info, label1_info

def compute_auc(score_file, mode='both', ignore_invalid=False):
    with open(score_file, 'r') as f:
        data = json.load(f)

    scores, labels, ids, label0, label1 = extract_per_image_scores(data, mode, ignore_invalid)
    if len(np.unique(labels)) < 2:
        print(f"[WARN] Only one class found in labels for mode={mode}, cannot compute AUC.")
        auc = None
    else:
        auc = roc_auc_score(labels, scores)

    return auc, {
        'all_scores': scores,
        'all_labels': labels,
        'all_ids': ids,
        'label0_ids': label0['ids'],
        'label0_scores': label0['scores'],
        'label1_ids': label1['ids'],
        'label1_scores': label1['scores']
    }

def eval_from_json(json_path, ignore_invalid=False):
    result_summary = {}
    for mode in ['ask_obj', 'ask_color', 'both']:
        auc, info = compute_auc(json_path, mode=mode, ignore_invalid=ignore_invalid)
        result_summary[mode] = {
            'auc': auc,
            'label0_num': len(info['label0_ids']),
            'label1_num': len(info['label1_ids']),
        }
    return result_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_json', type=str, required=True,
                        help='Path to result json from target_model_traverse.py')
    parser.add_argument('--ignore_invalid', action='store_true',
                        help='Ignore samples with invalid_info > 0')
    args = parser.parse_args()

    print(f"[INFO] Calculating AUCs (ignore_invalid={args.ignore_invalid})...")
    for mode in ['ask_obj', 'ask_color', 'both']:
        auc, info = compute_auc(args.res_json, mode=mode, ignore_invalid=args.ignore_invalid)
        if auc is not None:
            print(f"AUC ({mode}) = {auc:.4f} | label0: {len(info['label0_ids'])}, label1: {len(info['label1_ids'])}")
        else:
            print(f"AUC ({mode}) = N/A")

if __name__ == "__main__":
    main()
