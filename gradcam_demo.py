"""Regenerate the Grad-CAM visualization from the real trained ConvNeXt weights,
saved for the README (outputs/gradcam_example.png). ConvNeXt is used because it's a
conv-based model — Grad-CAM's assumptions (spatial feature maps + a gradient-weighted
class channel) apply directly; ViT needs a different technique (attention rollout)
since it has no spatial conv feature map to hook into.

    python gradcam_demo.py                  # picks an 'ig' example by default
    python gradcam_demo.py --class-name platelet --index 2
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import BloodMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model_utils import CLASSES, MODELS, device, load_model, transform, unnormalize

OUT = Path(__file__).parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-name", default="ig", choices=CLASSES,
                    help="which true class to pick an example from")
    ap.add_argument("--index", type=int, default=0,
                    help="which matching example to use, if class has several")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    name, weights = MODELS["ConvNeXt V2 Tiny"]
    model = load_model(name, weights)
    if model is None:
        raise SystemExit("ConvNeXt weights not found -- see README")

    test_ds = BloodMNIST(split='test', transform=transform, download=True)
    target_label = CLASSES.index(args.class_name)
    matches = [i for i in range(len(test_ds)) if int(np.array(test_ds.labels[i]).squeeze()) == target_label]
    idx = matches[args.index]

    img_tensor, label = test_ds[idx]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    label = int(np.array(label).squeeze())

    target_layers = [model.stages[-1].blocks[-1]]  # matches the notebook's Grad-CAM cell
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(label)])[0]

    img_display = unnormalize(img_tensor.cpu().squeeze(0)).permute(1, 2, 0).numpy()
    visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        pred = int(model(img_tensor).argmax(1).item())

    plt.figure(figsize=(4, 4.5))
    plt.imshow(visualization)
    plt.title(f"true: {CLASSES[label]}\npred: {CLASSES[pred]}", fontsize=10,
              color="green" if pred == label else "red")
    plt.axis("off")
    plt.tight_layout()
    out_path = OUT / "gradcam_example.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
