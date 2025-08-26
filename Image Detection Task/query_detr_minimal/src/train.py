import os
import argparse
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_coco import CocoSimpleDataset, collate_fn
from models.query_detector import QueryDetector
from utils.engine import train_one_epoch, evaluate
from utils.vis import save_samples

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-images", required=True, type=str)
    ap.add_argument("--val-images", required=True, type=str)
    ap.add_argument("--train-json", required=True, type=str)
    ap.add_argument("--val-json", required=True, type=str)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num-queries", type=int, default=12)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--image-size", nargs=2, type=int, default=[320,320])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    train_ds = CocoSimpleDataset(args.train_images, args.train_json, image_size=tuple(args.image_size))
    val_ds   = CocoSimpleDataset(args.val_images, args.val_json, image_size=tuple(args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = QueryDetector(num_classes=train_ds.num_classes, num_queries=args.num_queries, hidden_dim=args.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_hist = []
    val_hist = []

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        tr_ce, tr_l1 = train_one_epoch(model, train_loader, optimizer, device)
        va_ce, va_l1 = evaluate(model, val_loader, device)
        tr_total = tr_ce + 5.0 * tr_l1
        va_total = va_ce + 5.0 * va_l1
        train_hist.append(tr_total)
        val_hist.append(va_total)
        print(f" train: CE={tr_ce:.4f} L1={tr_l1:.4f} total={tr_total:.4f}")
        print(f"   val: CE={va_ce:.4f} L1={va_l1:.4f} total={va_total:.4f}")

        
        sample_dir = os.path.join(args.outdir, "samples")
        idxs = list(range(min(6, len(val_ds))))
        save_samples(model, val_ds, idxs, sample_dir, val_ds.label_to_name, device=device, score_thr=0.3)

    
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_hist, label="train")
    plt.plot(range(1, args.epochs+1), val_hist, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curves.png"), dpi=200)

    
    torch.save({"model": model.state_dict(), "num_classes": train_ds.num_classes,
                "num_queries": args.num_queries, "hidden_dim": args.hidden_dim},
               os.path.join(args.outdir, "model_final.pt"))
    print("Training complete. Outputs saved in:", args.outdir)

if __name__ == "__main__":
    main()
