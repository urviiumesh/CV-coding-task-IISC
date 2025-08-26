import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _class_cost(pred_logits, tgt_labels):
    # pred_logits: (T, K+1)
    # tgt_labels: (G,) each in [1..K]
    # cost: (T,G) = -log P(class=y_j)
    probs = F.softmax(pred_logits, dim=-1)  # (T,K+1)
    T = probs.shape[0]
    G = tgt_labels.shape[0]
    cost = torch.zeros((T,G), dtype=torch.float32, device=pred_logits.device)
    for j in range(G):
        lab = tgt_labels[j].long()
        cost[:, j] = -torch.log(torch.clamp(probs[:, lab], min=1e-8))
    return cost

def _box_l1_cost(pred_boxes, tgt_boxes):
    # pred_boxes: (T,4), tgt_boxes: (G,4)
    # pairwise L1
    T = pred_boxes.shape[0]
    G = tgt_boxes.shape[0]
    pb = pred_boxes.unsqueeze(1).repeat(1,G,1)  # (T,G,4)
    tb = tgt_boxes.unsqueeze(0).repeat(T,1,1)   # (T,G,4)
    return torch.abs(pb - tb).sum(-1)           # (T,G)

def hungarian_match(pred_logits, pred_boxes, tgt_labels, tgt_boxes, w_cls=1.0, w_box=5.0):
    # Returns index tensors (rows, cols) for matches. Empty if no targets.
    device = pred_logits.device
    G = tgt_boxes.shape[0]
    if G == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    c_cost = _class_cost(pred_logits, tgt_labels)  # (T,G)
    b_cost = _box_l1_cost(pred_boxes, tgt_boxes)   # (T,G)
    total = w_cls * c_cost + w_box * b_cost        # (T,G)

    if _HAS_SCIPY:
        # move to cpu for SciPy
        import numpy as np
        cost_np = total.detach().cpu().numpy()
        r, c = linear_sum_assignment(cost_np)
        return torch.as_tensor(r, dtype=torch.long, device=device), torch.as_tensor(c, dtype=torch.long, device=device)
    else:
        # Greedy fallback: iteratively choose the min remaining pair
        T, G = total.shape
        used_r = set()
        used_c = set()
        pairs = []
        vals, idxs = torch.sort(total.flatten())
        for v, idx in zip(vals, idxs):
            r = (idx // G).item()
            c = (idx %  G).item()
            if r not in used_r and c not in used_c:
                used_r.add(r); used_c.add(c)
                pairs.append((r,c))
            if len(pairs) == min(T, G):
                break
        if len(pairs) == 0:
            return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        rr = torch.as_tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        cc = torch.as_tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        return rr, cc
