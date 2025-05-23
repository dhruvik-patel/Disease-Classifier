# -*- coding: utf-8 -*-
"""cursor-pytorch-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Qc6T9JklfO8C4BJaLlGh7uIFgc6ZrtH7
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 1. Data Loading and Preparation
# ===============================

# Path to the data directories
DATA_PATH = '/kaggle/input/data/'
IMAGE_DIRS = [os.path.join(DATA_PATH, f'images_{str(i).zfill(3)}/images') for i in range(1, 13)]

# Path to the metadata CSV file (assumes a Data_Entry_2017.csv exists)
# This file should contain image names and disease labels
METADATA_PATH = os.path.join(DATA_PATH, 'Data_Entry_2017.csv')

# Define the 14 diseases (labels)
LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
          'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
          'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
          'Pleural_Thickening', 'Hernia']

# Image parameters
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_data():
    """
    Load image metadata and create full path to images
    """
    # Load and preprocess the metadata
    df = pd.read_csv(METADATA_PATH)

    # Convert the 'Finding Labels' column to binary encoded columns for each disease
    for label in LABELS:
        df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0)

    # Create full path to images
    df['image_path'] = ''

    # Create mapping between image filename and full path
    image_paths = {}
    for image_dir in IMAGE_DIRS:
        if os.path.exists(image_dir):
            for file in os.listdir(image_dir):
                if file.endswith('.png'):
                    image_paths[file] = os.path.join(image_dir, file)

    # Assign full path to each image in the dataframe
    df['image_path'] = df['Image Index'].map(image_paths)

    # Remove rows with missing image paths
    df = df[df['image_path'] != '']

    return df

# 2. Data Preprocessing and Dataset Creation
# =========================================

# Custom Dataset for Chest X-rays
class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe: DataFrame containing image paths and labels
            transform: Optional transform to be applied on a sample
        """
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')

        # Extract labels (14 disease labels)
        labels = self.dataframe.iloc[idx][LABELS].values.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(labels)

def prepare_data(df):
    """
    Split the data into training, validation, and test sets
    """
    # Create train/val/test splits (70/15/15)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(valid_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, valid_df, test_df

def create_data_loaders(train_df, valid_df, test_df):
    """
    Create data loaders for training, validation, and test sets
    """
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased rotation
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Added affine transformations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Enhanced color jitter
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Added Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)  # Added random erasing for additional regularization
    ])

    # Only resize and normalize for validation and test
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ChestXrayDataset(train_df, transform=train_transform)
    valid_dataset = ChestXrayDataset(valid_df, transform=val_test_transform)
    test_dataset = ChestXrayDataset(test_df, transform=val_test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader, test_df  # Return test_df for later use

# 3. Model Architecture
# ====================

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
            nn.Dropout(0.6),  # Increased dropout from 0.5 to 0.6
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout from 0.3 to 0.5
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Sigmoid for multi-label classification
        )

        # Apply weight decay to convolutional layers to improve regularization
        for m in self.densenet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.densenet(x)
        return self.classifier(features)

def create_model():
    """
    Create and initialize the model
    """
    model = ChestXrayModel(num_classes=len(LABELS))
    model = model.to(device)
    return model

# 4. Training and Evaluation
# =========================

def load_saved_model(model_path):
    # Create a new model instance
    model = create_model()

    # Load the state dictionary
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model file {model_path} not found. Starting with a new model.")

    return model


def train_model(model, train_loader, valid_loader, num_epochs=5, start_epoch=0, resume_from_checkpoint=None, reduce_overfitting=False):
    """
    Train the model with early stopping

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        num_epochs: Total number of epochs to train
        start_epoch: Starting epoch (useful when resuming training)
        resume_from_checkpoint: Path to model checkpoint to resume from
        reduce_overfitting: Whether to apply anti-overfitting techniques

    Returns:
        Trained model and training history
    """
    # Define loss function and optimizer
    criterion = nn.BCELoss()

    # Add weight decay if reducing overfitting
    if reduce_overfitting:
        optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)  # Reduced learning rate and added weight decay
        print("Applied anti-overfitting settings: weight decay and reduced learning rate")
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load optimizer state if resuming training
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint")
            # If reducing overfitting, modify optimizer's learning rate and weight decay
            if reduce_overfitting:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.00005
                    param_group['weight_decay'] = 1e-4
                print("Updated optimizer with anti-overfitting settings")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)

    # Early stopping parameters
    best_val_loss = float('inf')

    # Adjust patience for early stopping if reducing overfitting
    patience = 8 if reduce_overfitting else 5
    counter = 0
    best_model_path = 'best_pytorch_model.pth'

    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': []
    }

    # Load history if resuming training
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        if 'history' in checkpoint:
            history = checkpoint['history']
            print("Loaded training history from checkpoint")
            # If resuming, get the best validation loss from history
            if 'val_loss' in history and history['val_loss']:
                best_val_loss = min(history['val_loss'])

    # Calculate the total number of epochs (including those already completed)
    total_epochs = start_epoch + num_epochs

    # Create progress bar for epochs
    epochs_pbar = tqdm(total=total_epochs, initial=start_epoch, desc="Training",
                position=0, leave=True, mininterval=0.1, maxinterval=10.0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    print(f"Starting training from epoch {start_epoch+1} to {total_epochs} (total {num_epochs} new epochs)")

    # Print anti-overfitting status
    if reduce_overfitting:
        print("Anti-overfitting measures enabled: reduced learning rate, weight decay, increased patience")

    for epoch in range(start_epoch, total_epochs):
        # Training phase
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        # Create a progress bar for batches within this epoch
        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                          position=1, leave=False, mininterval=0.1)

        # Process training batches
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()

            # If reducing overfitting, clip gradients to prevent extreme weight updates
            if reduce_overfitting:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track loss
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss

            # Store predictions and targets for AUC calculation
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

            # Update batch progress bar with current loss
            batch_pbar.set_postfix(loss=f"{batch_loss/inputs.size(0):.4f}")
            batch_pbar.update(1)

        batch_pbar.close()

        # Calculate average training loss and AUC
        train_loss = train_loss / len(train_loader.dataset)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds, average='macro')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        # Create a progress bar for validation batches
        val_pbar = tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/{total_epochs} [Valid]",
                        position=1, leave=False, mininterval=0.1)

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Track loss
                batch_loss = loss.item() * inputs.size(0)
                val_loss += batch_loss

                # Store predictions and targets for AUC calculation
                val_preds.append(outputs.detach().cpu().numpy())
                val_targets.append(targets.detach().cpu().numpy())

                # Update validation progress bar
                val_pbar.set_postfix(loss=f"{batch_loss/inputs.size(0):.4f}")
                val_pbar.update(1)

        val_pbar.close()

        # Calculate average validation loss and AUC
        val_loss = val_loss / len(valid_loader.dataset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds, average='macro')

        # Adjust learning rate
        scheduler.step(val_loss)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update epochs progress bar
        epochs_pbar.update(1)
        epochs_pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            vl_loss=f"{val_loss:.4f}",
            tr_auc=f"{train_auc:.4f}",
            vl_auc=f"{val_auc:.4f}",
            time=f"{epoch_time:.1f}s"
        )

        # Print summary for this epoch
        print(f'Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

        # Calculate training/validation gap to monitor overfitting
        train_val_gap = train_auc - val_auc
        if reduce_overfitting and train_val_gap > 0.05:
            print(f"WARNING: Training-validation gap is {train_val_gap:.4f}, indicating potential overfitting")

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model with additional information for resuming training
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'history': history
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved checkpoint at epoch {epoch+1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Save regular checkpoint every epoch
        checkpoint_path = f'checkpoint_epoch_{epoch+1}_{val_auc:.4f}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved regular checkpoint at epoch {epoch+1}")

    # Close progress bar
    epochs_pbar.close()

    # Load best model
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss {checkpoint['val_loss']:.4f}")

    return model, history

def fine_tune_model(model, train_loader, valid_loader, num_epochs=10, resume_from_checkpoint=None):
    """
    Fine-tune the model by unfreezing the base model

    Args:
        model: PyTorch model to fine-tune
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        num_epochs: Total number of epochs to train
        resume_from_checkpoint: Path to model checkpoint to resume from

    Returns:
        Fine-tuned model and training history
    """
    # Start epoch
    start_epoch = 0

    # Load checkpoint if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming fine-tuning from epoch {start_epoch}")

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Define loss function and optimizer with a lower learning rate
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)  # Added weight decay

    # Load optimizer state if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state for fine-tuning")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-7)

    # Load scheduler state if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state for fine-tuning")

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 8  # Increased patience from 5 to 8
    counter = 0
    best_model_path = 'best_fine_tuned_pytorch_model.pth'

    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': []
    }

    # Load history if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        if 'history' in checkpoint:
            history = checkpoint['history']
            print("Loaded fine-tuning history from checkpoint")
            # Get best validation loss from history
            if 'val_loss' in history and history['val_loss']:
                best_val_loss = min(history['val_loss'])

    # Calculate the total number of epochs (including those already completed)
    total_epochs = start_epoch + num_epochs

    # Create progress bar for epochs
    epochs_pbar = tqdm(total=total_epochs, initial=start_epoch, desc="Fine-tuning",
                position=0, leave=True, mininterval=0.1, maxinterval=10.0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    print(f"Starting fine-tuning from epoch {start_epoch+1} to {total_epochs} (total {num_epochs} new epochs)")

    for epoch in range(start_epoch, total_epochs):
        # Training phase
        epoch_start_time = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        # Create a progress bar for batches within this epoch
        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{total_epochs} [Train]",
                          position=1, leave=False, mininterval=0.1)

        # Process training batches
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track loss
            batch_loss = loss.item() * inputs.size(0)
            train_loss += batch_loss

            # Store predictions and targets for AUC calculation
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(targets.detach().cpu().numpy())

            # Update batch progress bar with current loss
            batch_pbar.set_postfix(loss=f"{batch_loss/inputs.size(0):.4f}")
            batch_pbar.update(1)

        batch_pbar.close()

        # Calculate average training loss and AUC
        train_loss = train_loss / len(train_loader.dataset)
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        train_auc = roc_auc_score(train_targets, train_preds, average='macro')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        # Create a progress bar for validation batches
        val_pbar = tqdm(total=len(valid_loader), desc=f"Epoch {epoch+1}/{total_epochs} [Valid]",
                        position=1, leave=False, mininterval=0.1)

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Track loss
                batch_loss = loss.item() * inputs.size(0)
                val_loss += batch_loss

                # Store predictions and targets for AUC calculation
                val_preds.append(outputs.detach().cpu().numpy())
                val_targets.append(targets.detach().cpu().numpy())

                # Update validation progress bar
                val_pbar.set_postfix(loss=f"{batch_loss/inputs.size(0):.4f}")
                val_pbar.update(1)

        val_pbar.close()

        # Calculate average validation loss and AUC
        val_loss = val_loss / len(valid_loader.dataset)
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds, average='macro')

        # Adjust learning rate
        scheduler.step(val_loss)

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Update epochs progress bar
        epochs_pbar.update(1)
        epochs_pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            vl_loss=f"{val_loss:.4f}",
            tr_auc=f"{train_auc:.4f}",
            vl_auc=f"{val_auc:.4f}",
            time=f"{epoch_time:.1f}s"
        )

        # Print summary for this epoch
        print(f'Fine-tuning Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s | '
              f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint with all necessary information
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc,
                'history': history
            }
            torch.save(checkpoint, best_model_path)
            print(f"Saved fine-tuning checkpoint at epoch {epoch+1}")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        checkpoint_path = f'fine_tune_{epoch+1}_{val_auc:.4f}.pth'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_auc': val_auc,
            'history': history
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved regular fine-tuning checkpoint at epoch {epoch+1}")

    # Close progress bar
    epochs_pbar.close()

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best fine-tuned model from epoch {checkpoint['epoch']+1} with validation loss {checkpoint['val_loss']:.4f}")

    return model, history

def evaluate_model(model, test_loader):
    """
    Evaluate the model performance on test data
    """
    print("\nEvaluating model on test set...")
    model.eval()
    all_preds = []
    all_targets = []

    # Create a single progress bar with better settings
    batch_count = len(test_loader)
    print(f"Processing {batch_count} test batches...")

    pbar = tqdm(test_loader, desc="Evaluating",
                mininterval=0.1, maxinterval=10.0,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if batch_idx % 20 == 0:  # Print progress every 20 batches
                print(f"  Test batch {batch_idx}/{batch_count}")

            inputs = inputs.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    print("\nCalculating ROC AUC for each disease...")
    roc_auc = {}
    for i, label in enumerate(LABELS):
        if i % 5 == 0:  # Print progress every 5 labels
            print(f"  Processing disease {i+1}/{len(LABELS)}: {label}")
        roc_auc[label] = roc_auc_score(all_targets[:, i], all_preds[:, i])


    # Calculate mean ROC AUC
    mean_roc_auc = np.mean(list(roc_auc.values()))

    # Print results
    print(f"\nMean ROC AUC: {mean_roc_auc:.4f}")
    print("\nROC AUC for each disease:")
    for label, score in roc_auc.items():
        print(f"{label}: {score:.4f}")

    return roc_auc

def plot_training_history(history):
    """
    Plot the training and validation loss and AUC
    """
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation loss
    ax1.plot(history['train_loss'])
    ax1.plot(history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')

    # Plot training & validation AUC
    ax2.plot(history['train_auc'])
    ax2.plot(history['val_auc'])
    ax2.set_title('Model AUC')
    ax2.set_ylabel('AUC')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 5. Predictions for New Images
# ============================

def predict_disease(model, image_path):
    """
    Make predictions for a new image
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    img_display = np.array(img)  # Save for display

    img_tensor = transform(img).unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)[0].cpu().numpy()

    # Create a dictionary of disease probabilities
    disease_probs = {}
    for i, label in enumerate(LABELS):
        disease_probs[label] = float(prediction[i])

    # Sort by probability (descending)
    sorted_probs = sorted(disease_probs.items(), key=lambda x: x[1], reverse=True)

    # Display results
    plt.figure(figsize=(12, 8))

    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(img_display)
    plt.title('Input X-ray')
    plt.axis('off')

    # Display the probabilities
    plt.subplot(1, 2, 2)
    diseases = [item[0] for item in sorted_probs]
    probs = [item[1] * 100 for item in sorted_probs]

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
    plt.savefig('prediction_results.png')
    plt.show()

    return sorted_probs

print("1. Loading data...")
df = load_data()

print("\n2. Preparing data...")
train_df, valid_df, test_df = prepare_data(df)
train_loader, valid_loader, test_loader, test_df = create_data_loaders(train_df, valid_df, test_df)

# # Set to True to resume training from a saved checkpoint
resume_training = True
# # Path to the checkpoint to resume from (if resume_training is True)
resume_checkpoint = '/kaggle/working/checkpoint_epoch_19_0.8344.pth'
train_epochs = 3

# Set to True to skip training and just load a saved model for prediction/evaluation
load_trained_model = False
# # Path to the trained model to load (if load_trained_model is True)
# trained_model_path = 'best_fine_tuned_model.pth'

if load_trained_model and os.path.exists(trained_model_path):
    print("\n3. Loading pre-trained model...")
    model = load_saved_model(trained_model_path)
else:
    print("\n3. Creating model...")
    model = create_model()

    if resume_training and resume_checkpoint and os.path.exists(resume_checkpoint):
        print("\n4. Resuming training from checkpoint...")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0

        # To continue training with anti-overfitting measures
        print("\nContinuing training with anti-overfitting measures...")
        model, history = train_model(model, train_loader, valid_loader,
                                   num_epochs=train_epochs,
                                   start_epoch=start_epoch,
                                   resume_from_checkpoint=resume_checkpoint,
                                   reduce_overfitting=True)  # Enable anti-overfitting
    else:
        print("\n4. Training New model...")
        model, history = train_model(model, train_loader, valid_loader, num_epochs=train_epochs)
    print("\n4.1. Plotting training history...")
    plot_training_history(history)

    # fine tune after 20 epochs
    # Check if we want to resume fine-tuning from a checkpoint
    # resume_fine_tuning = False
    # fine_tune_checkpoint = 'fine_tune_checkpoint_epoch_5.pth'

    # if resume_fine_tuning and os.path.exists(fine_tune_checkpoint):
    #     print("\n5. Resuming fine-tuning from checkpoint...")
    #     model, fine_tune_history = fine_tune_model(model, train_loader, valid_loader,
    #                                              num_epochs=2,
    #                                              resume_from_checkpoint=fine_tune_checkpoint)
    # else:
    #     print("\n5. Fine-tuning model...")
    #     model, fine_tune_history = fine_tune_model(model, train_loader, valid_loader, num_epochs=10)

    # Save training history plot
    # print("\n5.1. Plotting training history...")
    # plot_training_history(fine_tune_history)

print("\n7. Evaluating model...")
model = create_model()
checkpoint = torch.load('/kaggle/input/test2/pytorch/default/1/checkpoint_epoch_14_0.8371.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# roc_auc = evaluate_model(model, test_loader)

print("\n8. Making predictions for a sample image...")
# Replace with an actual image path for prediction
# sample_image_path = test_df.iloc[0]['image_path']
# print(test_df.iloc[0]['image_path'])
sample_image_path = "/kaggle/input/data/images_001/images/00000001_001.png"
predictions = predict_disease(model, sample_image_path)

print("\nPrediction results:")
for disease, probability in predictions:
    print(f"{disease}: {probability*100:.2f}% probability")

