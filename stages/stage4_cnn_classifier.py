'''
Created on 11 Dec 2025

@author: Dr. Mike
Stage 4: CNN Classifier for ARIMA Parameter Selection

Builds and trains a Convolutional Neural Network to predict optimal ARIMA parameters
for time series based on their encoded image representations.

Architecture:
1. Backbone: ResNet50 (pre-trained on ImageNet)
   - Transfer learning approach
   - Frozen feature extraction layers
   - Leverages pre-learned image patterns
   
2. Custom Classification Head:
   - Global average pooling
   - Dense layer (256 units) + ReLU + BatchNorm + Dropout
   - Dense layer (128 units) + ReLU + BatchNorm + Dropout
   - Output layer (softmax) for class probabilities
   
3. Optimization:
   - Loss: Cross-entropy
   - Optimizer: Adam (lr=0.001)
   - Training: Multiple batches with validation

Training Process:
- Input: 64x64x3 images from IFFA stage
- Labels: ARIMA order classes (e.g., (1,1,0), (1,1,1), (0,1,1))
- Output: Probability distribution over ARIMA classes

For inference:
- Predicts most likely ARIMA order for new time series
- Returns prediction confidence score

PyTorch implementation with GPU support via CUDA.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from config import Config
from tqdm import tqdm


class ResNetClassifier(nn.Module):
    """ResNet-based classifier for ARIMA parameter selection"""
    
    def __init__(self, n_classes: int):
        super(ResNetClassifier, self).__init__()
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=True)
        
        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Identity()
        
        # Add custom classification head
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output


class CNNClassifier:
    """CNN-based classifier for ARIMA parameter selection (PyTorch)"""
    
    def __init__(self, img_size: int = Config.IMG_SIZE, 
                 batch_size: int = Config.BATCH_SIZE):
        """Initialize CNN classifier"""
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.device = Config.DEVICE
        self.label_encoder = None
        self.order_to_label = {}
        self.label_to_order = {}
        self.optimizer = None
        self.criterion = None
    
    def build_model(self, n_classes: int) -> None:
        """Build CNN model"""
        self.model = ResNetClassifier(n_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=Config.LEARNING_RATE)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = Config.EPOCHS,
              validation_split: float = Config.VALIDATION_SPLIT) -> dict:
        """Train CNN classifier"""
        if self.model is None:
            n_classes = y.shape[1]
            self.build_model(n_classes)
        
        # Split data
        n_samples = len(X)
        n_train = int(n_samples * (1 - validation_split))
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train = X[train_indices].astype(np.float32)
        y_train = np.argmax(y[train_indices], axis=1)
        X_val = X[val_indices].astype(np.float32)
        y_val = np.argmax(y[val_indices], axis=1)
        
        # Transpose from NHWC to NCHW format for PyTorch
        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_val = np.transpose(X_val, (0, 3, 1, 2))
        
        # Convert to torch tensors
        X_train_tensor = torch.from_numpy(X_train).to(self.device)
        y_train_tensor = torch.from_numpy(y_train).long().to(self.device)
        X_val_tensor = torch.from_numpy(X_val).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).long().to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            train_acc = 0.0
            
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_acc += (pred == y_batch).float().mean().item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
                val_pred = val_outputs.argmax(dim=1)
                val_acc = (val_pred == y_val_tensor).float().mean().item()
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['val_loss'].append(val_loss.item())
            history['train_acc'].append(train_acc / len(train_loader))
            history['val_acc'].append(val_acc)
            
            self.model.train()
        
        return history
    
    def predict(self, X: np.ndarray) -> tuple:
        """Predict ARIMA order for images"""
        self.model.eval()
        
        # Transpose from NHWC to NCHW format for PyTorch. NumPy/TensorFlow use NHWC, PyTorch uses NCHW
        X = np.transpose(X, Config.NHWC_TO_NCHW)
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            best_indices = torch.argmax(probabilities, dim=1)
            confidences = probabilities[torch.arange(len(probabilities)), best_indices]
        
        predicted_orders = [self.label_to_order[idx.item()] for idx in best_indices]
        
        return predicted_orders, confidences.cpu().numpy()
