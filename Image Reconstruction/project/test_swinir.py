# test_swinir.py
# Load CIFAR-10, run one batch through the SwinIRBlock, print shapes, and plot sample before/after.

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from swinir import SwinIRBlock

print("Starting script...")

def imshow(ax, tensor_img, title=None):
    # tensor_img: (3, H, W) or (H, W, 3)
    if tensor_img.ndim == 3 and tensor_img.shape[0] == 3:
        img = tensor_img.permute(1, 2, 0).cpu().numpy()
    else:
        img = tensor_img.cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    ax.imshow(img)
    if title:
        ax.set_title(title)
    ax.axis('off')

def main():
    print("Entering main function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),  # scales to [0,1]
    ])

    print("Loading CIFAR-10 dataset...")
    cifar_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(cifar_train, batch_size=8, shuffle=True, num_workers=0)

    print("Getting first batch...")
    images, labels = next(iter(dataloader))  # images shape: (B, 3, 32, 32)
    print("Loaded batch shape:", images.shape)

    images = images.to(device)

    print("Creating model...")
    model = SwinIRBlock(in_chans=3,
                        embed_dim=32,
                        input_resolution=(32, 32),
                        num_heads=4,
                        window_size=8,
                        shift_size=4).to(device)

    model.eval()
    print("Running inference...")
    with torch.no_grad():
        outputs = model(images)

    print("Input tensor shape:", images.shape)
    print("Output tensor shape:", outputs.shape)

    inputs_cpu = images.cpu()
    outputs_cpu = outputs.cpu()

    print("Plotting results...")
    # plot first 4 input-output pairs
    n = 4
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for i in range(n):
        imshow(axes[i, 0], inputs_cpu[i], title="Input")
        imshow(axes[i, 1], outputs_cpu[i].clamp(0.0, 1.0), title="Output")
    plt.tight_layout()
    plt.show()
    print("Done!")

if __name__ == "__main__":
    print("Script running as main...")
    main()
else:
    print("Script imported as module...")
