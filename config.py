'''
Created on 11 Dec 2025

@author: Dr. Mike

Global Configuration for IFFA-TSCI-CNN Project

This module contains all configurable parameters for the IFFA-TSCI-CNN pipeline.
Modify these settings to adjust image encoding, ARIMA optimization, CNN training,
validation methods, and data generation parameters.

Key Parameters:
- IMG_SIZE: Resolution of encoded time series images (64x64)
- MAX_P, MAX_D, MAX_Q: ARIMA parameter search ranges
- BATCH_SIZE, EPOCHS, LEARNING_RATE: CNN training configuration
- VALIDATION_METHOD: 'tl' (Transfer Learning) or 'skfcv' (Stratified K-Fold)
- N_SAMPLES, TS_LENGTH: Synthetic data generation settings
- DEVICE: Automatically detects GPU availability
'''
import torch
import time


class Config:
    """Global configuration for IFFA-TSCI-CNN project"""
    
    # Image encoding parameters
    IMG_SIZE = 64
    N_BINS = 20
    RECURRENCE_THRESHOLD = 0.1
    
    # ARIMA optimization parameters
    MAX_P = 5
    MAX_D = 3
    MAX_Q = 5
    ARIMA_TEST_SIZE = 0.2
    
    # CNN training parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model selection validation
    VALIDATION_METHOD = 'skfcv'  # 'tl' for Transfer Learning, 'skfcv' for Stratified K-Fold
    N_SPLITS = 5
    
    # Data generation parameters
    N_SAMPLES = 144 #MAX_P * MAX_D * MAX_Q
    TS_LENGTH = 160
    RANDOM_SEED = int(time.time() * 1000) % (2**31)  # Keep it within int32 range
    
   # Permutation tuple to convert NHWC → NCHW for PyTorch
    NHWC_TO_NCHW = (0, 3, 1, 2)
