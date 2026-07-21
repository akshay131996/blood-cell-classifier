import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image

# Configuration
# This should match the best model from the Colab notebook
MODEL_NAME = "convnextv2_tiny" 
WEIGHTS_PATH = f"{MODEL_NAME}_blood_cells.pth"
NUM_CLASSES = 8 # BloodMNIST has 8 classes

# BloodMNIST classes
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
try:
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model weights. {e}")
    print("Ensure you have trained the model in Colab and downloaded the .pth file!")
    model = None

# Transform for 224x224
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image):
    if model is None:
        return {"Error: Model weights not loaded. Please train first.": 1.0}
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
    # Format output for Gradio (dictionary mapping class name to probability)
    return {CLASSES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="Blood Cell Classifier",
    description="Upload an image of a blood cell to classify its type. This model was trained on the BloodMNIST dataset using a ConvNeXt architecture.",
    examples=[] # Add example paths here if desired
)

if __name__ == "__main__":
    iface.launch()
