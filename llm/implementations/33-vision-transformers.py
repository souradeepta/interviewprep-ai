"""
Auto-generated from 33-vision-transformers.ipynb
See notebook for detailed explanations and outputs
"""


# ======================================================================
# # 33 Vision Transformers
# ## Learning Objectives
# 1. Understand core concepts of 33 vision transformers
# 2. Implement with production libraries
# ======================================================================

# ======================================================================
# ## Level 1: Basic Implementation
# ======================================================================

import torch
from torchvision.transforms import ToTensor
from PIL import Image

# Simple Vision Transformer from scratch
class SimpleViT:
    def __init__(self, image_size=224, patch_size=16, embedding_dim=768):
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embedding_dim = embedding_dim

        # Initialize learnable parameters
        self.class_token = torch.randn(1, 1, embedding_dim)
        self.patch_embeddings = torch.randn(self.num_patches, embedding_dim)
        self.position_embeddings = torch.randn(self.num_patches + 1, embedding_dim)

    def extract_patches(self, image):
        # Image: (C, H, W), output patches
        patches = []
        for i in range(0, image.shape[1], self.patch_size):
            for j in range(0, image.shape[2], self.patch_size):
                patch = image[:, i:i+self.patch_size, j:j+self.patch_size]
                patch = patch.flatten()  # Flatten to 1D
                patches.append(patch)
        return torch.stack(patches)  # (num_patches, patch_size*patch_size*C)

# Load and process image
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Simulate with random image
image = torch.randn(3, 224, 224)
vit = SimpleViT()

patches = vit.extract_patches(image)
print(f"Image shape: {image.shape}")
print(f"Number of patches: {patches.shape[0]}")
print(f"Patch embedding dim: {vit.embedding_dim}")



# ======================================================================
# ## Level 2: Advanced Implementation
# ======================================================================

import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests

class ViTClassifier:
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def classify(self, image_url):
        # Download and load image
        image = Image.open(requests.get(image_url, stream=True).raw)

        # Extract features
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get top-k predictions
        logits = outputs.logits
        top_k = torch.topk(logits, 5)

        return top_k

    def get_attention_maps(self, image_url):
        # Get attention from intermediate layers
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        attentions = outputs.attentions
        return attentions  # Tuple of attention maps per layer

# Initialize classifier
classifier = ViTClassifier()

# Example: classify ImageNet image
test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
print("ViT model initialized and ready for classification")
print(f"Model name: google/vit-base-patch16-224")
print(f"Number of patches: 196 (14x14 grid of 16x16 patches)")



# ======================================================================
# ## Real-World Example 1: Production Pattern
# ======================================================================

# Real-World: Fine-tune ViT on Custom Dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import Dataset, DataLoader
import torch

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, feature_extractor):
        self.images = images  # List of PIL Images
        self.labels = labels
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        inputs = self.feature_extractor(self.images[idx], return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(self.labels[idx])
        }

# Load pretrained ViT
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10  # 10 custom classes
)

# Create dummy dataset
num_images = 100
dummy_images = [torch.randint(0, 256, (3, 224, 224)).permute(1,2,0).numpy().astype('uint8')
                 for _ in range(num_images)]
dummy_labels = [i % 10 for i in range(num_images)]

dataset = CustomImageDataset(dummy_images, dummy_labels, feature_extractor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

print(f"Dataset size: {len(dataset)}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print("Ready for fine-tuning")



# ======================================================================
# ## Real-World Example 2: Advanced Usage
# ======================================================================

# Real-World: Visualize ViT Attention
import matplotlib.pyplot as plt
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

def visualize_attention(image_url, layer=0, head=0):
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", output_attentions=True)

    # Load and process image
    from PIL import Image
    import requests
    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Get attention maps
    with torch.no_grad():
        outputs = model(**inputs)
        attention = outputs.attentions[layer]  # (batch, heads, seq_len, seq_len)

    # Attention for specific head
    att_map = attention[0, head, 0, 1:].view(14, 14)  # CLS token attention to patches

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(att_map.numpy(), cmap='hot')
    plt.title(f"Attention Map (Layer {layer}, Head {head})")

    plt.show()

print("ViT attention visualization code ready")
print("Attention reveals which patches the model focuses on for classification")



# ======================================================================
# ## Real-World Example 3: Optimization
# ======================================================================

# Real-World: ViT for Zero-Shot Classification
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

class ViTZeroShot:
    def __init__(self):
        # Note: Zero-shot requires CLIP-like model, here using ViT for illustration
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    def classify_new_categories(self, image, categories):
        '''
        Zero-shot: classify into categories not in training data
        Approach: Use embedding similarity instead of softmax
        '''
        # Get image embedding
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embedding = outputs.logits  # Use last layer as embedding

        # In practice, would use text encoder for categories and compare embeddings
        # Here showing the pattern
        print(f"Image embedding: {image_embedding.shape}")
        print("Zero-shot requires dual encoders (image + text)")
        return image_embedding

# For true zero-shot, use CLIP
from transformers import CLIPProcessor, CLIPModel

class CLIPZeroShot:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def classify(self, image, categories):
        # Encode image and text
        inputs = self.processor(text=categories, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_categories)

        scores = torch.softmax(logits_per_image, dim=1)[0]
        for category, score in zip(categories, scores):
            print(f"{category}: {score:.3f}")

print("CLIP zero-shot classification pattern:")
print("1. Encode image with vision model")
print("2. Encode categories with text model")
print("3. Compute similarity (logits)")
print("4. Apply softmax to get probabilities")




# ======================================================================
# ## Key Takeaways
# **When to use 33 vision transformers:**
# - Understand when this concept applies
# - Consider tradeoffs and constraints
# ======================================================================
