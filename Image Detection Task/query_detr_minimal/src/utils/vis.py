import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

def draw_predictions_on_image(pil_img, pred_logits, pred_boxes, label_to_name, score_thr=0.3):
    
    probs = torch.softmax(pred_logits, dim=-1)
    scores, labels = probs.max(-1)
    keep = (labels != 0) & (scores >= score_thr)
    draw = ImageDraw.Draw(pil_img)
    W, H = pil_img.size

    for i in torch.nonzero(keep).flatten().tolist():
        cx, cy, w, h = pred_boxes[i].tolist()
        x0 = int((cx - w/2) * W)
        y0 = int((cy - h/2) * H)
        x1 = int((cx + w/2) * W)
        y1 = int((cy + h/2) * H)
        lab = labels[i].item()
        score = scores[i].item()
        name = label_to_name.get(lab, str(lab))
        draw.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=2)
        draw.text((x0, max(0,y0-12)), f"{name}:{score:.2f}", fill=(255,255,255))

    return pil_img

def save_samples(model, dataset, indices, outdir, label_to_name, device, score_thr=0.3):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    for idx in indices:
        img, target = dataset[idx]
        pil = transforms_to_pil(img)
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))
        pred_logits = out["pred_logits"][0].cpu()
        pred_boxes  = out["pred_boxes"][0].cpu()
        pil = draw_predictions_on_image(pil, pred_logits, pred_boxes, label_to_name, score_thr=score_thr)
        fn = target.get("file_name", f"sample_{idx}.png")
        pil.save(os.path.join(outdir, f"pred_{os.path.basename(fn)}"))

def transforms_to_pil(tensor_img):
    
    from torchvision import transforms
    
    inv = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[1/0.229,1/0.224,1/0.225]),
        transforms.Normalize(mean=[-0.485,-0.456,-0.406], std=[1,1,1]),
        transforms.ToPILImage()
    ])
    try:
        return inv(tensor_img.cpu().clamp(0,1))
    except Exception:
        return transforms.ToPILImage()(tensor_img.cpu().clamp(0,1))
