import json
import argparse
from sklearn.metrics import roc_auc_score

def compute_auc(grouped_data, question_type, ignore_invalid):
    scores = []
    labels = []
    label0_ids = []
    label1_ids = []

    for entry in grouped_data:
        label = entry["ground_truth_label"]
        img_id = entry["img_id"]

        # select only questions of target type
        target_questions = [
            q for q in entry["questions"]
            if q["question_type"] == question_type and (not ignore_invalid or q["invalid_info"] == 0)
        ]

        if not target_questions:
            continue

        image_score = sum(q["acc"] for q in target_questions) / len(target_questions)
        scores.append(image_score)
        labels.append(label)

        if label == 0:
            label0_ids.append(img_id)
        else:
            label1_ids.append(img_id)

    if len(set(labels)) < 2:
        print("[WARN] Only one label class found, AUC cannot be computed.")
        return None, len(label0_ids), len(label1_ids)

    auc = roc_auc_score(labels, scores)
    return auc, len(label0_ids), len(label1_ids)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_json', type=str, required=True)
    parser.add_argument('--question_type', type=str, required=True)
    parser.add_argument('--ignore_invalid', action='store_true')
    args = parser.parse_args()

    with open(args.result_json, 'r') as f:
        grouped_data = json.load(f)

    auc, n0, n1 = compute_auc(grouped_data, args.question_type, args.ignore_invalid)

    if auc is not None:
        print(f"[AUC] {auc:.4f} | label0: {n0}, label1: {n1}")
    else:
        print("[AUC] N/A")

if __name__ == "__main__":
    main()