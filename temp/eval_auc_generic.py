
import json
import argparse
from sklearn.metrics import roc_auc_score

def compute_auc(results, ignore_flag):
    scores, labels = [], []
    label0_info = {"scores": [], "ids": []}
    label1_info = {"scores": [], "ids": []}
    skipped = 0

    for entry in results:
        label = entry["ground_truth_label"]
        img_id = entry["img_id"]
        
        accs = []
        for q in entry["questions"]:
            if ignore_flag and q["invalid_info"] > 0:
                continue
            accs.append(q["acc"])
        
        if not accs:
            skipped += 1  # 如果某张图片的所有问题都 Invalid, 在 ignore_flag=True 时会忽略该图片 
            continue

        avg_score = sum(accs) / len(accs)
        scores.append(avg_score)
        labels.append(label)

        if label == 0:
            label0_info["scores"].append(avg_score)
            label0_info["ids"].append(img_id)
        else:
            label1_info["scores"].append(avg_score)
            label1_info["ids"].append(img_id)
    
    if len(set(labels)) < 2:
        print(f"[WARN] Only one class found in labels for ignore_flag={ignore_flag}, cannot compute AUC.")
        auc = None
    else:
        auc = roc_auc_score(labels, scores)

    return {
        "auc": auc,
        "skipped": skipped,
        "label0": label0_info,
        "label1": label1_info,
        "used": len(scores),
        "total": len(results)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_json', type=str, required=True)
    parser.add_argument('--ignore_flag', action='store_true')
    args = parser.parse_args()

    with open(args.result_json, 'r') as f:
        results = json.load(f)

    result = compute_auc(results, args.ignore_flag)
    print(f"[AUC] {result['auc']:.4f}")
    print(f"[INFO] label0: {len(result['label0']['ids'])}, label1: {len(result['label1']['ids'])}")
    print(f"[INFO] skipped: {result['skipped']}, used: {result['used']} / {result['total']}")

if __name__ == "__main__":
    main()
