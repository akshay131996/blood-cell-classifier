import gradio as gr
import torch
import timm
from torchvision import transforms

# Two backbones, side by side — this IS the project's story (CNN vs transformer).
MODELS = {
    "ConvNeXt V2 Tiny": ("convnextv2_tiny", "convnextv2_tiny_blood_cells.pth"),
    "ViT-Small/16": ("vit_small_patch16_224", "vit_small_patch16_224_blood_cells.pth"),
}
NUM_CLASSES = 8
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte',
           'monocyte', 'neutrophil', 'platelet']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


loaded = {label: load_model(name, path) for label, (name, path) in MODELS.items()}


def predict(image):
    if image is None:
        return {}, {}
    img_tensor = transform(image).unsqueeze(0).to(device)
    outputs = []
    for label in MODELS:
        model = loaded[label]
        if model is None:
            outputs.append({f"[{label} weights not found — see README]": 1.0})
            continue
        with torch.no_grad():
            probs = torch.nn.functional.softmax(model(img_tensor)[0], dim=0)
        outputs.append({CLASSES[i]: float(probs[i]) for i in range(NUM_CLASSES)})
    return outputs[0], outputs[1]


with gr.Blocks(title="Blood Cell Classifier — ConvNeXt vs ViT") as iface:
    gr.Markdown(
        "# 🩸 Blood Cell Classifier\n"
        "Upload a blood-cell microscopy image (BloodMNIST classes) and compare predictions "
        "from two fine-tuned backbones side by side."
    )
    with gr.Row():
        image_in = gr.Image(type="pil", label="Cell image")
        with gr.Column():
            out_convnext = gr.Label(num_top_classes=3, label="ConvNeXt V2 Tiny")
            out_vit = gr.Label(num_top_classes=3, label="ViT-Small/16")
    image_in.change(predict, inputs=image_in, outputs=[out_convnext, out_vit])

if __name__ == "__main__":
    iface.launch()
