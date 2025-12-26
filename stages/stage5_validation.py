'''
Created on 11 Dec 2025

@author: Dr. Mike
'''
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
from config import Config
from .stage4_cnn_classifier import CNNClassifier


class TransferLearningValidation:
    """Transfer Learning validation scheme"""
    
    def __init__(self, cnn_classifier):
        self.cnn_classifier = cnn_classifier
        self.history = None
    
    def validate(self, X: np.ndarray, y: np.ndarray, 
                 test_size: float = Config.ARIMA_TEST_SIZE) -> dict:
        """Validate using simple train-test split"""
        n = len(X)
        n_train = int(n * (1 - test_size))
        
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print("    Training on transfer learning split...")
        self.history = self.cnn_classifier.train(X_train, y_train, 
                                                 epochs=Config.EPOCHS)
        
        # Evaluate
        import torch
        self.cnn_classifier.model.eval()
        X_test_transposed = np.transpose(X_test, (0, 3, 1, 2))
        X_test_tensor = torch.from_numpy(X_test_transposed.astype(np.float32)).to(self.cnn_classifier.device)
        y_test_tensor = torch.from_numpy(np.argmax(y_test, axis=1)).long().to(self.cnn_classifier.device)
        
        with torch.no_grad():
            outputs = self.cnn_classifier.model(X_test_tensor)
            loss = self.cnn_classifier.criterion(outputs, y_test_tensor).item()
            pred = outputs.argmax(dim=1)
            accuracy = (pred == y_test_tensor).float().mean().item()
        
        return {
            'method': 'Transfer Learning (TL)',
            'test_accuracy': accuracy,
            'test_loss': loss
        }


class StratifiedKFoldValidation:
    """Stratified K-Fold Cross Validation scheme"""
    
    def __init__(self, cnn_classifier):
        self.cnn_classifier = cnn_classifier
    
    def validate(self, X: np.ndarray, y: np.ndarray, 
                 n_splits: int = Config.N_SPLITS) -> dict:
        """Validate using Stratified K-Fold Cross Validation"""
        y_labels = np.argmax(y, axis=1)
        
        # Adjust n_splits if needed based on class sizes
        min_class_count = np.bincount(y_labels).min()
        actual_splits = min(n_splits, min_class_count)
        
        # Ensure at least 2 splits
        actual_splits = max(2, actual_splits)
        
        if actual_splits < n_splits:
            print(f"    Note: Reducing folds from {n_splits} to {actual_splits} (min class count: {min_class_count})")
        
        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, 
                             random_state=Config.RANDOM_SEED)
        
        fold_accuracies = []
        fold_losses = []
        
        print(f"    Performing Stratified {actual_splits}-Fold Cross Validation...")
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_labels), 1):
            print(f"      Fold {fold}/{actual_splits}...", end=' ')
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            classifier = CNNClassifier(self.cnn_classifier.img_size, 
                                      self.cnn_classifier.batch_size)
            classifier.label_to_order = self.cnn_classifier.label_to_order
            classifier.train(X_train, y_train, epochs=Config.EPOCHS, 
                           validation_split=0.0)
            
            import torch
            classifier.model.eval()
            X_test_transposed = np.transpose(X_test, (0, 3, 1, 2))
            X_test_tensor = torch.from_numpy(X_test_transposed.astype(np.float32)).to(classifier.device)
            y_test_tensor = torch.from_numpy(np.argmax(y_test, axis=1)).long().to(classifier.device)
            
            with torch.no_grad():
                outputs = classifier.model(X_test_tensor)
                loss = classifier.criterion(outputs, y_test_tensor).item()
                pred = outputs.argmax(dim=1)
                accuracy = (pred == y_test_tensor).float().mean().item()
            
            fold_accuracies.append(accuracy)
            fold_losses.append(loss)
            
            print(f"Accuracy: {accuracy:.4f}")
        
        return {
            'method': f'Stratified K-Fold Cross Validation (S-FCV, k={actual_splits})',
            'fold_accuracies': np.array(fold_accuracies),
            'fold_losses': np.array(fold_losses),
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies),
            'mean_loss': np.mean(fold_losses)
        }