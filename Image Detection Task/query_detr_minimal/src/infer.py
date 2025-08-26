import os
import argparse
import torch
from torch.utils.data import DataLoader

from dataset_coco import CocoSimpleDataset, collate_fn
from models.query_detector import QueryDetector
from utils.vis import save_samples

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=str)
    ap.add_argument("--json", required=True, type=str)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--outdir", type=str, default="outputs/infer_samples")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--image-size", nargs=2, type=int, default=[320,320])
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    ds = CocoSimpleDataset(args.images, args.json, image_size=tuple(args.image_size))

    ckpt = torch.load(args.ckpt, map_location=device)
    model = QueryDetector(num_classes=ckpt["num_classes"], num_queries=ckpt["num_queries"], hidden_dim=ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["model"])

    idxs = list(range(min(20, len(ds))))
    save_samples(model, ds, idxs, args.outdir, ds.label_to_name, device=device, score_thr=args.score_thr)
    print("Saved samples to", args.outdir)

if __name__ == "__main__":
    main()
