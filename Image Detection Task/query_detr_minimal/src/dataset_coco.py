import os, json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CocoSimpleDataset(Dataset):
    def __init__(self, images_dir, coco_json, image_size=(320,320), normalize=True):
        self.images_dir = images_dir
        with open(coco_json, "r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.anns_by_img = {}
        for a in coco["annotations"]:
            self.anns_by_img.setdefault(a["image_id"], []).append(a)
        cats = coco["categories"]
        # labels start at 1..K to reserve 0 for no-object
        self.catid_to_label = {c["id"]: i+1 for i,c in enumerate(cats)}
        self.label_to_catid = {v:k for k,v in self.catid_to_label.items()}
        self.label_to_name = {i+1: c["name"] for i,c in enumerate(cats)}
        self.num_classes = len(cats)
        self.image_size = tuple(image_size)

        t = [transforms.ToTensor()]
        if normalize:
            t.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
        self.transform = transforms.Compose(t)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        path = os.path.join(self.images_dir, info["file_name"])
        img = Image.open(path).convert("RGB")
        W, H = img.size
        img_resized = img.resize(self.image_size)

        tensor = self.transform(img_resized)

        anns = self.anns_by_img.get(info["id"], [])
        boxes = []
        labels = []
        for a in anns:
            x,y,w,h = a["bbox"]
            cx = (x + 0.5*w) / W
            cy = (y + 0.5*h) / H
            nw = w / W
            nh = h / H
            boxes.append([cx,cy,nw,nh])
            labels.append(self.catid_to_label[a["category_id"]])

        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes,
            "labels": labels,
            "orig_size": (H, W),
            "file_name": info["file_name"],
        }
        return tensor, target

def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
