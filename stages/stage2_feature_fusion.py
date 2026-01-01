'''
Created on 11 Dec 2025

@author: Dr. Mike
Stage 2: Information Fusion Feature Augmentation (IFFA) - Basic Version

Intelligently combines the three time series images (GAF, MTF, RP) using
information fusion techniques and feature augmentation.

Process:
1. Image Fusion:
   - Stacks GAF, MTF, RP images together
   - Applies weighted averaging with fixed weights (0.4, 0.35, 0.25)
   - Creates a single fused representation capturing all three perspectives
   
2. Feature Augmentation:
   - Channel 1: Original fused image
   - Channel 2: Smoothed version (local averaging)
   - Channel 3: Edge-enhanced version (edge detection)
   
Output: 64x64x3 augmented image ready for CNN input

This is the BASIC version with fixed weights. For learnable fusion weights
and more sophisticated feature extraction, see stage2_feature_optimal_fusion.py
'''
import numpy as np
from scipy.signal import convolve2d
from config import Config

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'statsmodels'])
    from statsmodels.tsa.arima.model import ARIMA
    
class InformationFusionFeatureAugmentation:
    """Fuses information from multiple time series images"""
    
    def __init__(self, weights: np.ndarray = None):
        """Initialize IFFA with fusion weights"""
        if weights is None:
            self.weights = np.array([0.4, 0.35, 0.25])
        else:
            self.weights = weights
    
    def fuse_images(self, gaf: np.ndarray, mtf: np.ndarray, rp: np.ndarray) -> np.ndarray:
        """Fuse three image representations through weighted combination"""
        stacked = np.stack([gaf, mtf, rp], axis=-1)
        fused = np.average(stacked, axis=-1, weights=self.weights)
        return fused
    
    def augment_features(self, fused: np.ndarray) -> np.ndarray:
        """Create augmented features through preprocessing"""
        channel1 = fused
        
        kernel = np.ones((3, 3)) / 9
        channel2 = convolve2d(fused, kernel, mode='same')
        # Edge detection kernel: 8-neighbor Laplacian operator
        # This is a well-established edge detection filter, not arbitrary numbers!
        # 
        # How it works:
        # - Center value (+8): weights the current pixel
        # - Surrounding values (-1 each): weights the 8 neighboring pixels
        # - Sum = 0: makes it a "high-pass filter" that responds to changes
        # 
        # When convolved over the image:
        # - Flat/uniform regions → output ≈ 0 (no edge)
        # - Rapid intensity changes → large output (edge detected!)
        # 
        # Mathematical basis: discrete approximation of the 2D Laplacian (∇²f)
        # which computes the second derivative to find areas of rapid change.
        # 
        # Alternative kernels you could try:
        # - Laplacian 4-neighbor: [[0,-1,0], [-1,8,-1], [0,-1,0]] (only horizontal/vertical)
        # - Sobel: Detects edges in specific directions (X or Y)
        edge_kernel = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]])
        channel3 = convolve2d(fused, edge_kernel, mode='same')
        channel3 = (channel3 - channel3.min()) / (channel3.max() - channel3.min() + 1e-8)
        
        augmented = np.stack([channel1, channel2, channel3], axis=-1)
        return augmented
    
    def process(self, gaf: np.ndarray, mtf: np.ndarray, rp: np.ndarray) -> np.ndarray:
        """Complete IFFA pipeline"""
        fused = self.fuse_images(gaf, mtf, rp)
        augmented = self.augment_features(fused)
        return augmented
