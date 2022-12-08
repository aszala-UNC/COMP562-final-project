from train import test
import torch
import torchvision.transforms as transforms
from data.dataset import DATA_DIR
from data.tokenizer import Tokenizer
import language_evaluation
import json
from pprint import PrettyPrinter
pprint = PrettyPrinter().pprint


if __name__ == "__main__":
    with open(f"{DATA_DIR}/test.json", 'r') as f:
        gt = json.load(f)

        gt_values = { }

        for d in gt:
            gt_values[d["id"]] = d["caption"]

    CHECKPOINT_PATH = "./checkpoints/resent_no_dropout"

    NUM_WORKERS = 4

    MAX_WORD_LENGTH = 32

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PREPROCESS = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = Tokenizer()
    tokenizer.load(f"{DATA_DIR}/vocab.txt")

    VOCAB_SIZE = tokenizer.vocab_size

    pred_values = test(f"{CHECKPOINT_PATH}/best.pth", tokenizer, preprocess=PREPROCESS, device=DEVICE, max_length=MAX_WORD_LENGTH, num_workers=NUM_WORKERS)

    print(pred_values)

    gt_list = []
    pred_list = []

    out = {  }

    for key in pred_values:
        pred = pred_values[key]
        gt = gt_values[key]

        pred_list.append(pred)
        gt_list.append(gt)

        out[key] = { "predicted": pred, "truth": gt }

    evaluator = language_evaluation.CocoEvaluator()
    results = evaluator.run_evaluation(pred_list, gt_list)
    pprint(results)

    out = {
        "score_results": results,
        "full_output_comparison": out
    }

    with open(f"./results/{CHECKPOINT_PATH.split('/')[-1]}.json", 'w') as f:
        json.dump(out, f, indent=2)
