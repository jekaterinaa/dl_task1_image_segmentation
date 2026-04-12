import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import torch
from torchvision import transforms


def colorize_mask(mask):
    """
    mask: (H, W) with values 0..3
    returns: (H, W, 3) uint8 RGB image with bright colors
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    colors = {
        0: (0, 0, 0),        # background - black
        1: (255, 0, 0),      # Cat - red
        2: (0, 255, 0),      # Dog - green
        3: (0, 0, 255),      # Car - blue
    }

    for c, col in colors.items():
        rgb[mask == c] = col

    return rgb


def visualize_predictions(model, img_dir, mask_dir, device, num_samples=3, img_size=(256, 256)):
    model.eval()

    img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if len(img_files) == 0:
        print("No .jpg images found in test directory.")
        return

    selected = random.sample(img_files, min(num_samples, len(img_files)))

    to_tensor = transforms.ToTensor()

    for img_name in selected:
        img_path = os.path.join(img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        img_orig = Image.open(img_path).convert("RGB")

        img_resized = img_orig.resize(img_size, Image.BILINEAR)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {img_name}, skipping.")
            continue
        gt_mask_pil = Image.open(mask_path).resize(img_size, Image.NEAREST)
        gt_mask = np.array(gt_mask_pil, dtype=np.int64)
        gt_mask = gt_mask // 80
        gt_mask = np.clip(gt_mask, 0, 3)

        inp = to_tensor(img_resized).unsqueeze(0).to(device)  # (1,3,H,W)
        with torch.no_grad():
            logits = model(inp)
            preds = torch.argmax(logits, dim=1)  # (1,H,W)

        pred_mask = preds.squeeze(0).cpu().numpy().astype(np.int64)

        gt_colored = colorize_mask(gt_mask)
        pred_colored = colorize_mask(pred_mask)

        plt.figure(figsize=(12, 4))
        plt.suptitle(img_name)

        plt.subplot(1, 3, 1)
        plt.imshow(img_orig)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_colored)
        plt.title("GT mask (colored)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_colored)
        plt.title("Pred mask (colored)")
        plt.axis("off")

        plt.show()


def preprocess_web_image(img_path, img_size=(256, 256)):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float).unsqueeze(0)
