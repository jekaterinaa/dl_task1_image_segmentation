import torch
import torch.nn.functional as functional
import numpy as np

def get_pixel_predictions(model_torch, class_names, img_tensor, x, y):
    model_torch.eval()
    with torch.no_grad():
        logits = model_torch(img_tensor)  # (1,C,H,W)
        probs = functional.softmax(logits, dim=1)  # (1,C,H,W)

    C = logits.shape[1]
    H = logits.shape[2]
    W = logits.shape[3]

    assert 0 <= x < W and 0 <= y < H, f"(x,y)=({x},{y}) outside image size (W,H)=({W},{H})"

    pixel_logits = logits[0, :, y, x].detach().cpu()  # (C,)
    pixel_probs  = probs[0, :, y, x].detach().cpu()   # (C,)

    pred_class = int(torch.argmax(pixel_probs).item())
    pred_name  = class_names.get(pred_class, str(pred_class))

    print(f"Tensor size: (H,W)=({H},{W})  |  queried pixel (x,y)=({x},{y})\n")

    print("Per-class logits at pixel:")
    for c in range(C):
        print(f"  class {c:>1} ({class_names.get(c,'?'):>10}): {pixel_logits[c].item(): .4f}")

    print("\nPer-class probabilities at pixel:")
    for c in range(C):
        print(f"  class {c:>1} ({class_names.get(c,'?'):>10}): {pixel_probs[c].item(): .4%}")

    print(f"\nPredicted class at (x={x}, y={y}): {pred_class} ({pred_name})")