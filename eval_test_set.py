"""Evaluate both trained backbones on BloodMNIST's official held-out test split.

The training run only logged train/val accuracy — this closes that gap with a real
generalization number the val accuracy could theoretically have been optimistic about.

    python eval_test_set.py
"""
import torch
from medmnist import BloodMNIST
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from model_utils import CLASSES, MODELS, device, load_model, transform


def main():
    test_ds = BloodMNIST(split='test', transform=transform, download=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"test set: {len(test_ds)} images\n")

    for label, (timm_name, weights_path) in MODELS.items():
        model = load_model(timm_name, weights_path)
        if model is None:
            print(f"{label}: skipped\n")
            continue

        preds, truth = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                out = model(images.to(device))
                preds += out.argmax(1).cpu().tolist()
                truth += labels.squeeze().tolist()

        acc = sum(p == t for p, t in zip(preds, truth)) / len(truth)
        print(f"== {label}: test accuracy = {acc:.4f} ==")
        print(classification_report(truth, preds, target_names=CLASSES, digits=3))
        print("confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(truth, preds))
        print()


if __name__ == "__main__":
    main()
