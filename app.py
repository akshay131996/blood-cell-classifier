import gradio as gr
import torch

from model_utils import CLASSES, MODELS, NUM_CLASSES, load_all, transform

loaded = load_all()


def predict(image):
    if image is None:
        return {}, {}
    img_tensor = transform(image).unsqueeze(0)
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
