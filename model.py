import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent Tkinter errors
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define the 14 diseases (labels)
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
          'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
          'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
          'Pleural_Thickening', 'Hernia']

# Image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model architecture
class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXrayModel, self).__init__()

        # Load pre-trained DenseNet121
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_features = self.densenet.classifier.in_features

        # Replace the classifier with custom layers for multi-label classification
        self.densenet.classifier = nn.Identity()

        # Add custom classification layers with increased dropout rates
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.densenet(x)
        return self.classifier(features)

def load_model(model_path):
    """
    Load a trained model from a checkpoint file
    """
    model = ChestXrayModel(num_classes=len(LABELS))
    model = model.to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        try:
            # First try loading with weights_only=False (only for trusted models)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            print("Model loaded successfully with weights_only=False")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Trying alternative loading method...")
            
            # If the first attempt fails, try a different approach
            try:
                from torch.serialization import add_safe_globals
                import numpy as np
                # Add numpy scalar to safe globals
                add_safe_globals([np.core.multiarray.scalar])
                checkpoint = torch.load(model_path, map_location=device)
                print("Model loaded successfully with added safe globals")
            except Exception as inner_e:
                raise RuntimeError(f"Failed to load model: {str(inner_e)}")
                
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")

def predict_image(model, image):
    """
    Make predictions for an image
    
    Args:
        model: Trained PyTorch model
        image: PIL Image object
        
    Returns:
        List of (disease, probability) tuples sorted by probability
    """
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Transform and add batch dimension
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)[0].cpu().numpy()
    
    # Create a dictionary of disease probabilities
    disease_probs = {}
    for i, label in enumerate(LABELS):
        disease_probs[label] = float(prediction[i])
    
    # Sort by probability (descending)
    sorted_probs = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_probs

def generate_prediction_plot(image, predictions):
    """
    Generate a plot showing the input image and disease probabilities
    
    Args:
        image: PIL Image
        predictions: List of (disease, probability) tuples
        
    Returns:
        Base64 encoded string of the plot image
    """
    plt.figure(figsize=(12, 8))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Input X-ray')
    plt.axis('off')
    
    # Display the probabilities
    plt.subplot(1, 2, 2)
    diseases = [item[0] for item in predictions]
    probs = [item[1] * 100 for item in predictions]
    
    colors = ['#1f77b4' if p > 50 else '#d62728' for p in probs]
    
    y_pos = np.arange(len(diseases))
    plt.barh(y_pos, probs, color=colors)
    plt.yticks(y_pos, diseases)
    plt.xlabel('Probability (%)')
    plt.title('Disease Probabilities')
    plt.xlim(0, 100)
    
    for i, prob in enumerate(probs):
        plt.text(prob + 1, i, f'{prob:.1f}%', va='center')
    
    plt.tight_layout()
    
    # Save plot to a BytesIO object
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Encode to base64 string
    encoded = base64.b64encode(image_png).decode('utf-8')
    
    plt.close()
    
    return encoded 