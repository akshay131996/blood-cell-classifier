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

<!-- Fill these in after training — a table like this is what employers look for. -->

| Model | Params | Val accuracy | Test accuracy |
|---|---|---|---|
| ConvNeXt V2 Tiny | ~28M | _TODO_ | _TODO_ |
| ViT-Small/16 | ~22M | _TODO_ | _TODO_ |

_Add a Grad-CAM example image and a confusion matrix here once trained._

## Notes

- Model weights (`*.pth`) are gitignored — regenerate them from the notebook.
- Dataset: [BloodMNIST](https://medmnist.com/) (Yang et al., MedMNIST v2).
