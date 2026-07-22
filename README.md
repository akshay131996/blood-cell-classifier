# Blood Cell Classifier 🩸

Fine-grained classification of white/red blood cells from microscopy images, built with
modern transfer learning. Trains and compares two backbones — a modernized CNN
(**ConvNeXt V2**) and a Vision Transformer (**ViT-Small**) — on the **BloodMNIST**
dataset (8 cell types), with a Gradio demo for live inference.

> Part of my computer-vision portfolio. This project focuses on the 2026 fine-tuning
> workflow: pretrained backbones via `timm`, mixed-precision training, experiment
> tracking with W&B, and interpretability with Grad-CAM.

## The 8 classes

`basophil` · `eosinophil` · `erythroblast` · `immature granulocytes (ig)` ·
`lymphocyte` · `monocyte` · `neutrophil` · `platelet`

## What's here

| File | Purpose |
|---|---|
| `cv_project_1_biology_classifier.ipynb` | End-to-end training notebook (data → train ConvNeXt + ViT → Grad-CAM) |
| `app.py` | Gradio demo — upload a cell image, get top-3 predictions |
| `gen_notebook.py` | Script that generates the notebook |
| `requirements.txt` | Dependencies |

## Approach

- **Data:** BloodMNIST via the official [`medmnist`](https://medmnist.com/) package, upscaled to 224×224.
- **Models:** `convnextv2_tiny` and `vit_small_patch16_224` from `timm`, pretrained on ImageNet, fine-tuned on blood cells — a direct CNN-vs-transformer comparison.
- **Training:** AdamW, cross-entropy, mixed-precision (`torch.amp`), augmentation (flips + rotations). Metrics logged to Weights & Biases.
- **Interpretability:** Grad-CAM heatmaps to confirm the model attends to the cell, not the background — important for any medical-imaging model.

## Run it

```bash
pip install -r requirements.txt

# 1) Train (best on a GPU — Google Colab or Kaggle). Open the notebook:
#    cv_project_1_biology_classifier.ipynb  -> produces convnextv2_tiny_blood_cells.pth

# 2) Launch the demo (uses the trained .pth):
python app.py
```

## Results

5 epochs, AdamW, mixed precision, on BloodMNIST (train/val split). Full run history and
loss curves are logged to W&B — [ConvNeXt run](https://wandb.ai/akshayanil4-none/cv-sprint-blood-cells/runs/mmyezgvs) ·
[ViT run](https://wandb.ai/akshayanil4-none/cv-sprint-blood-cells/runs/kvfvz0k1).

| Model | Params | Train accuracy | Val accuracy (best epoch) |
|---|---|---|---|
| ConvNeXt V2 Tiny | 27.9M | 97.8% | **96.5%** (epoch 4) |
| ViT-Small/16 | 21.7M | 98.1% | **97.3%** (epoch 3) |

**Takeaway:** ViT-Small edged out ConvNeXt on validation accuracy (97.3% vs 96.5%) despite
having ~6M fewer parameters. The common wisdom is that ViTs need more data than CNNs to
close the inductive-bias gap — but BloodMNIST's ~12k training images and centered,
low-clutter cell images were apparently enough for a small ImageNet-pretrained ViT to
fine-tune well. Both models show mild overfitting by epoch 5 (train accuracy ~1.5-2pp
above the best val accuracy) — more augmentation or early stopping would likely close
that gap further.

**Honest gap:** only train/val were logged in this run, no separate held-out test-set
evaluation. Before calling this "done," the natural next step is scoring both checkpoints
on BloodMNIST's official test split (via `medmnist`) to confirm val accuracy wasn't
optimistic.

Below: Grad-CAM on an immature granulocyte — the heatmap centers on the cell body itself
rather than background, which is the interpretability bar for any medical-imaging model.

![Grad-CAM example](outputs/gradcam_example.png)

## Notes

- Model weights (`*.pth`) are gitignored — regenerate them from the notebook.
- Dataset: [BloodMNIST](https://medmnist.com/) (Yang et al., MedMNIST v2).
