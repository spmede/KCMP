import os
import json
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def extract_color_exist_scores(result_data, ignore_invalid=False):
    scores = []
    labels = []
    label0_info = {'ids': [], 'scores': []}
    label1_info = {'ids': [], 'scores': []}

    for entry in result_data:
        label = entry['label']
        accs = []

        for q in entry.get('queries', []):
            if ignore_invalid and q.get('invalid_info', 0) > 0:
                continue
            accs.append(q.get('acc', 0))

        if accs:
            mean_acc = np.mean(accs)
            scores.append(mean_acc)
            labels.append(label)
            if label == 0:
                label0_info['ids'].append(entry['img_id'])
                label0_info['scores'].append(mean_acc)
            else:
                label1_info['ids'].append(entry['img_id'])
                label1_info['scores'].append(mean_acc)

    return np.array(scores), np.array(labels), label0_info, label1_info

def compute_color_auc(file_path, ignore_invalid):
    with open(file_path, 'r') as f:
        data = json.load(f)

    scores, labels, label0_info, label1_info = extract_color_exist_scores(data, ignore_invalid=ignore_invalid)

    if len(np.unique(labels)) < 2:
        print("[WARN] Only one label class found, AUC cannot be computed.")
        return None, label0_info, label1_info

    auc = roc_auc_score(labels, scores)
    return auc, label0_info, label1_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_json', type=str, required=True)
    parser.add_argument('--ignore_invalid', action='store_true')
    args = parser.parse_args()

    print(f"[INFO] Evaluating Color Exist AUCs (ignore_invalid={args.ignore_invalid})...")
    auc, label0_info, label1_info = compute_color_auc(args.res_json, args.ignore_invalid)

    if auc is not None:
        print(f"AUC = {auc:.4f} | label0: {len(label0_info['ids'])}, label1: {len(label1_info['ids'])}")
    else:
        print("AUC = N/A")

if __name__ == '__main__':
    main()
