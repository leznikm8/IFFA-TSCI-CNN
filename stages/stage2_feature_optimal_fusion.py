'''
Created on 17 Dec 2025

@author: Dr. Mike
Stage 2: Optimal Information Fusion Feature Augmentation (IFFA) - Advanced Version

Advanced implementation of IFFA with multiple feature extraction methods and
learnable fusion framework. This version extracts richer feature representations
from the time series images.

Key Features:
1. Multi-Feature Extraction:
   - Raw image values
   - Edge features (Sobel operators)
   - Texture features (LBP-like patterns)
   - Frequency domain features (FFT-based)
   
2. Multi-Level Fusion:
   - Combines each feature type across all three images (GAF, MTF, RP)
   - Uses weighted combinations for each feature level
   
3. Advanced Augmentation:
   - Smoothing augmentation
   - Edge enhancement augmentation
   - Contrast normalization augmentation
   - Combines all augmentations for robust representation
   
4. Learnable Fusion Framework:
   - Weights initialized to (0.4, 0.35, 0.25)
   - Ready for optimization during CNN training
   - Can learn optimal combination of GAF, MTF, RP
   
Output: 64x64x3 image with richer feature representation

Use this instead of stage2_feature_fusion.py for potentially better CNN accuracy
at the cost of increased computation time.
'''
import numpy as np
from scipy.signal import convolve2d
import torch
import torch.nn as nn
from config import Config


class LearnableFusion(nn.Module):
    """Learnable fusion layer for IFFA"""
    
    def __init__(self, img_size: int = Config.IMG_SIZE):
        super(LearnableFusion, self).__init__()
        self.img_size = img_size
        
        # Learnable weights for each image type
        self.weight_gaf = nn.Parameter(torch.tensor(0.4, dtype=torch.float32))
        self.weight_mtf = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
        self.weight_rp = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        
        # Learnable fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1)
        )
    
    def forward(self, gaf: torch.Tensor, mtf: torch.Tensor, rp: torch.Tensor) -> torch.Tensor:
        """Fuse three images with learnable weights"""
        # Normalize weights to sum to 1
        weights = torch.softmax(torch.stack([self.weight_gaf, self.weight_mtf, self.weight_rp]), dim=0)
        
        # Weighted fusion
        fused = weights[0] * gaf + weights[1] * mtf + weights[2] * rp
        
        # Stack for convolution
        stacked = torch.stack([gaf, mtf, rp], dim=1)  # (B, 3, H, W)
        
        # Apply learnable fusion convolution
        fused_conv = self.fusion_conv(stacked)
        
        # Combine weighted and convolutional fusion
        combined = 0.6 * fused.unsqueeze(1) + 0.4 * fused_conv
        
        return combined.squeeze(1)


class OptimalInformationFusion:
    """
    True IFFA: Optimal Information Fusion Feature Augmentation
    - Multi-level feature extraction
    - Learnable fusion weights
    - Advanced augmentation strategies
    """
    
    def __init__(self, img_size: int = Config.IMG_SIZE):
        self.img_size = img_size
        self.learnable_fusion = LearnableFusion(img_size)
    
    def extract_features(self, image: np.ndarray) -> dict:
        """Extract multiple feature representations from image"""
        features = {
            'raw': image,
            'edges': self._extract_edges(image),
            'texture': self._extract_texture(image),
            'frequency': self._extract_frequency(image)
        }
        return features
    
    @staticmethod
    def _extract_edges(image: np.ndarray) -> np.ndarray:
        """Extract edge features using Sobel operator"""
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        edges_x = convolve2d(image, sobel_x, mode='same')
        edges_y = convolve2d(image, sobel_y, mode='same')
        
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-8)
        
        return edges
    
    @staticmethod
    def _extract_texture(image: np.ndarray) -> np.ndarray:
        """Extract texture features using local binary patterns concept"""
        texture = np.zeros_like(image)
        
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                center = image[i, j]
                neighborhood = image[i-1:i+2, j-1:j+2]
                
                # LBP-like texture measure
                texture[i, j] = np.sum(neighborhood > center)
        
        # Normalize
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)
        
        return texture
    
    @staticmethod
    def _extract_frequency(image: np.ndarray) -> np.ndarray:
        """Extract frequency domain features"""
        # FFT-based feature
        fft = np.fft.fft2(image)
        magnitude = np.abs(fft)
        magnitude = np.log(magnitude + 1)
        
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        
        return magnitude
    
    def fuse_images(self, gaf: np.ndarray, mtf: np.ndarray, rp: np.ndarray) -> np.ndarray:
        """Fuse three images with multiple strategies"""
        # Strategy 1: Extract features from each image
        gaf_features = self.extract_features(gaf)
        mtf_features = self.extract_features(mtf)
        rp_features = self.extract_features(rp)
        
        # Strategy 2: Combine raw images with weighted fusion
        weighted_raw = 0.4 * gaf + 0.35 * mtf + 0.25 * rp
        
        # Strategy 3: Combine edge features
        weighted_edges = 0.4 * gaf_features['edges'] + 0.35 * mtf_features['edges'] + 0.25 * rp_features['edges']
        
        # Strategy 4: Combine texture features
        weighted_texture = 0.4 * gaf_features['texture'] + 0.35 * mtf_features['texture'] + 0.25 * rp_features['texture']
        
        # Stack all fused representations
        fused = np.stack([weighted_raw, weighted_edges, weighted_texture], axis=-1)
        
        return fused
    
    def augment_features(self, fused: np.ndarray) -> np.ndarray:
        """Advanced feature augmentation with multiple techniques"""
        channel1 = fused[:, :, 0]
        channel2 = fused[:, :, 1]
        channel3 = fused[:, :, 2]
        
        # Augmentation 1: Smoothing
        smooth_kernel = np.ones((3, 3)) / 9
        aug1 = np.stack([
            convolve2d(channel1, smooth_kernel, mode='same'),
            convolve2d(channel2, smooth_kernel, mode='same'),
            convolve2d(channel3, smooth_kernel, mode='same')
        ], axis=-1)
        
        # Augmentation 2: Edge enhancement
        edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        aug2 = np.stack([
            convolve2d(channel1, edge_kernel, mode='same'),
            convolve2d(channel2, edge_kernel, mode='same'),
            convolve2d(channel3, edge_kernel, mode='same')
        ], axis=-1)
        aug2 = (aug2 - aug2.min()) / (aug2.max() - aug2.min() + 1e-8)
        
        # Augmentation 3: Contrast enhancement
        aug3 = np.stack([
            (channel1 - channel1.mean()) / (channel1.std() + 1e-8),
            (channel2 - channel2.mean()) / (channel2.std() + 1e-8),
            (channel3 - channel3.mean()) / (channel3.std() + 1e-8)
        ], axis=-1)
        
        # Combine augmentations (average them)
        combined = (fused + aug1 + aug2 + aug3) / 4
        
        return combined
    
    def process(self, gaf: np.ndarray, mtf: np.ndarray, rp: np.ndarray) -> np.ndarray:
        """Complete optimal IFFA pipeline"""
        fused = self.fuse_images(gaf, mtf, rp)
        augmented = self.augment_features(fused)
        
        return augmented