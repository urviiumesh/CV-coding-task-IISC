import torch
import torch.nn.functional as F
from tqdm import tqdm
from .matcher import hungarian_match

def compute_losses_for_image(pred_logits, pred_boxes, tgt_labels, tgt_boxes, box_weight=5.0):
    device = pred_logits.device
    T = pred_logits.shape[0]
    
    cls_target = torch.zeros((T,), dtype=torch.long, device=device)

    if tgt_boxes.numel() == 0:
        ce = F.cross_entropy(pred_logits, cls_target)
        l1 = torch.tensor(0.0, device=device)
        return ce, l1

    rows, cols = hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes)

    
    cls_target[rows] = tgt_labels[cols].long().to(device)

    ce = F.cross_entropy(pred_logits, cls_target)

    if rows.numel() > 0:
        l1 = F.l1_loss(pred_boxes[rows], tgt_boxes[cols].to(device), reduction='mean')
    else:
        l1 = torch.tensor(0.0, device=device)

    return ce, l1

def train_one_epoch(model, loader, optimizer, device, box_weight=5.0):
    model.train()
    sum_ce, sum_l1, n = 0.0, 0.0, 0
    for images, targets in tqdm(loader, desc="train"):
        images = torch.stack(images).to(device)      
        outputs = model(images)
        logits = outputs["pred_logits"]              
        boxes  = outputs["pred_boxes"]               

        optimizer.zero_grad()
        ce_total = 0.0
        l1_total = 0.0
        B = logits.shape[0]
        for b in range(B):
            tl = targets[b]["labels"].to(device)
            tb = targets[b]["boxes"].to(device)
            ce, l1 = compute_losses_for_image(logits[b], boxes[b], tl, tb, box_weight=box_weight)
            ce_total = ce_total + ce
            l1_total = l1_total + l1

        loss = ce_total + box_weight * l1_total
        loss.backward()
        optimizer.step()

        sum_ce += ce_total.item()
        sum_l1 += l1_total.item()
        n += B

    return sum_ce / max(1,n), sum_l1 / max(1,n)

@torch.no_grad()
def evaluate(model, loader, device, box_weight=5.0):
    model.eval()
    sum_ce, sum_l1, n = 0.0, 0.0, 0
    for images, targets in tqdm(loader, desc="val"):
        images = torch.stack(images).to(device)
        outputs = model(images)
        logits = outputs["pred_logits"]
        boxes  = outputs["pred_boxes"]
        B = logits.shape[0]
        for b in range(B):
            tl = targets[b]["labels"].to(device)
            tb = targets[b]["boxes"].to(device)
            ce, l1 = compute_losses_for_image(logits[b], boxes[b], tl, tb, box_weight=box_weight)
            sum_ce += ce.item()
            sum_l1 += l1.item()
            n += 1
    return sum_ce / max(1,n), sum_l1 / max(1,n)
