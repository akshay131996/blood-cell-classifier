"""Shared model loading + preprocessing — used by app.py, eval_test_set.py,
gradcam_demo.py, and view_predictions.py so they all agree on how to load
weights and preprocess images.
"""
import torch
import timm
from torchvision import transforms

MODELS = {
    "ConvNeXt V2 Tiny": ("convnextv2_tiny", "convnextv2_tiny_blood_cells.pth"),
    "ViT-Small/16": ("vit_small_patch16_224", "vit_small_patch16_224_blood_cells.pth"),
}
NUM_CLASSES = 8
# Official BloodMNIST label order (medmnist.INFO['bloodmnist']['label']) --
# 'ig' is short for 'immature granulocytes (myelocytes, metamyelocytes, promyelocytes)'.
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte',
           'monocyte', 'neutrophil', 'platelet']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


def load_model(timm_name, weights_path):
    try:
        m = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
        m.load_state_dict(torch.load(weights_path, map_location=device))
        m.to(device).eval()
        print(f"loaded {weights_path}")
        return m
    except Exception as e:
        print(f"could not load {weights_path}: {e}\n"
              f"  -> train in the notebook, download the .pth, and place it next to app.py")
        return None


def load_all():
    return {label: load_model(name, path) for label, (name, path) in MODELS.items()}
