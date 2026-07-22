"""View N test-set predictions at once: image, true label, prediction, confidence.
The fastest way to eyeball whether a model is actually working, not just accurate on
paper -- errors cluster in visually-similar classes far more often than they scatter
randomly, and that pattern only shows up when you look at a grid like this.

    python view_predictions.py                          # 8 random ConvNeXt predictions
    python view_predictions.py --model "ViT-Small/16" --n 12 --seed 1
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import BloodMNIST

from model_utils import CLASSES, MODELS, device, load_model, transform, unnormalize

OUT = Path(__file__).parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ConvNeXt V2 Tiny", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    name, weights = MODELS[args.model]
    model = load_model(name, weights)
    if model is None:
        raise SystemExit(f"{args.model} weights not found -- see README")

    test_ds = BloodMNIST(split='test', transform=transform, download=True)
    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(test_ds), size=args.n, replace=False)

    cols = 4
    rows = -(-args.n // cols)  # ceil
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax, idx in zip(np.array(axes).flat, idxs):
        img_tensor, label = test_ds[idx]
        label = int(np.array(label).squeeze())
        with torch.no_grad():
            probs = torch.softmax(model(img_tensor.unsqueeze(0).to(device))[0], dim=0)
        pred = int(probs.argmax())
        conf = float(probs[pred])

        img = unnormalize(img_tensor).permute(1, 2, 0).numpy()
        ax.imshow(img)
        correct = pred == label
        ax.set_title(f"true: {CLASSES[label]}\npred: {CLASSES[pred]} ({conf:.0%})",
                     fontsize=8, color="green" if correct else "red")
        ax.axis("off")
    for ax in np.array(axes).flat[len(idxs):]:
        ax.axis("off")

    fig.suptitle(f"{args.model} — {args.n} random test predictions", fontsize=11)
    fig.tight_layout()
    out_path = OUT / "predictions_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
