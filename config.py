'''
Created on 11 Dec 2025

@author: Dr. Mike
'''
import torch

class Config:
    """Global configuration for IFFA-TSCI-CNN project"""
    
    # Image encoding parameters
    IMG_SIZE = 64
    N_BINS = 20
    RECURRENCE_THRESHOLD = 0.1
    
    # ARIMA optimization parameters
    MAX_P = 3
    MAX_D = 2
    MAX_Q = 3
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
    N_SAMPLES = 30
    TS_LENGTH = 150
    RANDOM_SEED = 11
