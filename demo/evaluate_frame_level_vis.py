import json
import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser("eval frame-level vis dataset")
    parser.add_argument("--task", default="segm", type=str, help="bbox or segm or key points")
    parser.add_argument("--gt_json_file", default="datasets/ytvis_2021/val_sub_frame_level.json",
                        type=str, help="ground-truth json file")
    parser.add_argument("--pred_json_file", default="output/ytvis_2021/results_valid_sub_frame_level.json",
                        type=str, help="the json file of predicted results")
    return parser.parse_args()

def evaluate_frame_level_ytvis(gt_json_file, pred_json_file, task="segm"):
    gt_annos = COCO(gt_json_file)
    pred_results = json.load(open(pred_json_file, "r"))

    dt = gt_annos.loadRes(pred_results)
    eval = COCOeval(gt_annos, dt, iouType=task)

    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return eval

if __name__ == "__main__":
    args = parse_args()

    evaluate_frame_level_ytvis(args.gt_json_file, args.pred_json_file, args.task)