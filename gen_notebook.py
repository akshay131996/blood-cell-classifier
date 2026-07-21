import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Project 1: Fine-grained Biology Classifier (Blood Cells)\n",
    "\n",
    "This notebook trains two models (`convnext_v2_tiny` and `vit_small_patch16`) on a blood cell image dataset to compare their performance. We will use `timm` for the models and `wandb` for tracking."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install -q timm wandb gradio medmnist grad-cam"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader\n",
    "from torchvision import transforms\n",
    "import torchvision\n",
    "import timm\n",
    "import wandb\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from tqdm.auto import tqdm\n",
    "import medmnist\n",
    "from medmnist import INFO\n",
    "\n",
    "# Ensure GPU is available\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Preparation\n",
    "We use the official `medmnist` package to download and load the BloodMNIST dataset cleanly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_flag = 'bloodmnist'\n",
    "info = INFO[data_flag]\n",
    "DataClass = getattr(medmnist, info['python_class'])\n",
    "\n",
    "# Get the class names from the official metadata\n",
    "classes = list(info['label'].values())\n",
    "num_classes = len(classes)\n",
    "print(f\"Classes ({num_classes}): {classes}\")\n",
    "\n",
    "# Define transforms for 224x224 models\n",
    "train_transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.RandomHorizontalFlip(),\n",
    "    transforms.RandomRotation(15),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "val_transform = transforms.Compose([\n",
    "    transforms.Resize((224, 224)),\n",
    "    transforms.ToTensor(),\n",
    "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "])\n",
    "\n",
    "# Download and load the datasets\n",
    "train_ds = DataClass(split='train', transform=train_transform, download=True)\n",
    "val_ds = DataClass(split='val', transform=val_transform, download=True)\n",
    "test_ds = DataClass(split='test', transform=val_transform, download=True)\n",
    "\n",
    "batch_size = 64\n",
    "train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)\n",
    "val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)\n",
    "test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)\n",
    "\n",
    "print(f\"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Visualize Sample Images\n",
    "Let's look at a batch of the training data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Grab a single batch of training data\n",
    "sample_images, sample_labels = next(iter(train_loader))\n",
    "\n",
    "# Un-normalize the images to display them correctly\n",
    "def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):\n",
    "    mean = torch.tensor(mean).view(3, 1, 1)\n",
    "    std = torch.tensor(std).view(3, 1, 1)\n",
    "    return tensor * std + mean\n",
    "\n",
    "# Select the first 8 images to display\n",
    "num_images = 8\n",
    "subset_images = sample_images[:num_images]\n",
    "subset_labels = sample_labels[:num_images].squeeze()\n",
    "\n",
    "# Un-normalize\n",
    "unnorm_images = unnormalize(subset_images)\n",
    "\n",
    "# Create a grid\n",
    "grid = torchvision.utils.make_grid(unnorm_images, nrow=4, padding=2)\n",
    "np_grid = grid.numpy()\n",
    "np_grid = np.transpose(np_grid, (1, 2, 0)) # Convert from (C, H, W) to (H, W, C)\n",
    "np_grid = np.clip(np_grid, 0, 1) # Ensure values are between 0 and 1\n",
    "\n",
    "# Plot\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.imshow(np_grid)\n",
    "plt.axis('off')\n",
    "plt.title(\"Sample Blood Cell Images\")\n",
    "plt.show()\n",
    "\n",
    "# Print the labels\n",
    "print(\"Labels:\", [classes[label.item()] for label in subset_labels])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Training Loop Setup\n",
    "Generic training function using mixed precision (`torch.amp.autocast`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_model(model_name, epochs=5):\n",
    "    wandb.init(project=\"cv-sprint-blood-cells\", name=model_name)\n",
    "    \n",
    "    # Load pretrained model\n",
    "    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)\n",
    "    model = model.to(device)\n",
    "    \n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = optim.AdamW(model.parameters(), lr=1e-4)\n",
    "    scaler = torch.amp.GradScaler('cuda')\n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "        model.train()\n",
    "        train_loss, train_correct = 0, 0\n",
    "        \n",
    "        for images, labels in tqdm(train_loader, desc=f\"Epoch {epoch+1}/{epochs} [Train]\"):\n",
    "            images = images.to(device)\n",
    "            labels = labels.squeeze().to(device)\n",
    "            \n",
    "            optimizer.zero_grad()\n",
    "            with torch.amp.autocast('cuda'):\n",
    "                outputs = model(images)\n",
    "                loss = criterion(outputs, labels)\n",
    "                \n",
    "            scaler.scale(loss).backward()\n",
    "            scaler.step(optimizer)\n",
    "            scaler.update()\n",
    "            \n",
    "            train_loss += loss.item() * images.size(0)\n",
    "            train_correct += (outputs.argmax(1) == labels).sum().item()\n",
    "            \n",
    "        # Validation\n",
    "        model.eval()\n",
    "        val_loss, val_correct = 0, 0\n",
    "        with torch.no_grad():\n",
    "            for images, labels in tqdm(val_loader, desc=f\"Epoch {epoch+1}/{epochs} [Val]\"):\n",
    "                images = images.to(device)\n",
    "                labels = labels.squeeze().to(device)\n",
    "                \n",
    "                with torch.amp.autocast('cuda'):\n",
    "                    outputs = model(images)\n",
    "                    loss = criterion(outputs, labels)\n",
    "                    \n",
    "                val_loss += loss.item() * images.size(0)\n",
    "                val_correct += (outputs.argmax(1) == labels).sum().item()\n",
    "                \n",
    "        epoch_train_loss = train_loss / len(train_ds)\n",
    "        epoch_train_acc = train_correct / len(train_ds)\n",
    "        epoch_val_loss = val_loss / len(val_ds)\n",
    "        epoch_val_acc = val_correct / len(val_ds)\n",
    "        \n",
    "        wandb.log({\n",
    "            \"train_loss\": epoch_train_loss, \n",
    "            \"train_acc\": epoch_train_acc, \n",
    "            \"val_loss\": epoch_val_loss, \n",
    "            \"val_acc\": epoch_val_acc\n",
    "        })\n",
    "        print(f\"Val Acc: {epoch_val_acc:.4f} | Val Loss: {epoch_val_loss:.4f}\")\n",
    "        \n",
    "    # Save the model\n",
    "    torch.save(model.state_dict(), f\"{model_name}_blood_cells.pth\")\n",
    "    wandb.finish()\n",
    "    return model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Train Models\n",
    "Login to wandb, then train ConvNeXt and ViT."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import wandb\n",
    "wandb.login() # This will prompt for your API key"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training ConvNeXt...\")\n",
    "model_convnext = train_model('convnextv2_tiny', epochs=5)\n",
    "\n",
    "print(\"\\nTraining ViT...\")\n",
    "model_vit = train_model('vit_small_patch16_224', epochs=5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Grad-CAM / Interpretability (Optional)\n",
    "We can use `pytorch-grad-cam` to visualize where the model is looking."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pytorch_grad_cam import GradCAM\n",
    "from pytorch_grad_cam.utils.image import show_cam_on_image\n",
    "from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget\n",
    "\n",
    "# Example for ConvNeXt\n",
    "target_layers = [model_convnext.stages[-1].blocks[-1]]\n",
    "cam = GradCAM(model=model_convnext, target_layers=target_layers)\n",
    "\n",
    "sample_images, sample_labels = next(iter(test_loader))\n",
    "img_tensor = sample_images[0].unsqueeze(0).to(device)\n",
    "label = sample_labels[0].item()\n",
    "\n",
    "targets = [ClassifierOutputTarget(label)]\n",
    "grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]\n",
    "\n",
    "# Convert back to image for display\n",
    "unnorm_img = unnormalize(img_tensor.cpu().squeeze(0)).numpy()\n",
    "img_display = np.transpose(unnorm_img, (1, 2, 0))\n",
    "img_display = np.clip(img_display, 0, 1)\n",
    "\n",
    "visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)\n",
    "\n",
    "plt.imshow(visualization)\n",
    "plt.title(f\"True Label: {classes[label]}\")\n",
    "plt.axis('off')\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("cv_project_1_biology_classifier.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
