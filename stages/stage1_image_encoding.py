'''
Created on 11 Dec 2025

@author: Dr. Mike
Stage 1: Time Series Image Encoding

Converts 1D time series data into three complementary 2D image representations:

1. Gramian Angular Field (GAF):
   - Converts time series values to polar coordinates
   - Creates angular cosine distance matrix
   - Captures magnitude relationships and patterns
   
2. Markov Transition Field (MTF):
   - Quantizes time series into discrete states
   - Records transition probabilities between states
   - Captures temporal dynamics and state changes
   
3. Recurrence Plot (RP):
   - Calculates pairwise distance matrix
   - Binary representation of recurrence patterns
   - Shows dynamical behavior in phase space

Each output is a 64x64 numpy array representing the time series as an image,
enabling the use of computer vision techniques for time series analysis.
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from config import Config



class GramianAngularField:
    """Convert time series to Gramian Angular Field (GAF)"""
    
    @staticmethod
    def encode(ts: np.ndarray, img_size: int = Config.IMG_SIZE) -> np.ndarray:
        """Transform time series to GAF"""
        # Ensure input is numpy array
        ts = np.asarray(ts, dtype=np.float64).flatten()
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        ts_norm = scaler.fit_transform(ts.reshape(-1, 1)).flatten()
        
        n = len(ts_norm)
        theta = np.arccos(np.clip(ts_norm, -1, 1))
        
        gaf = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                gaf[i, j] = np.cos(theta[i] + theta[j])
        
        # Resize using scipy
        zoom_factor = img_size / n
        gaf_resized = ndimage.zoom(gaf, zoom_factor, order=1)
        gaf_resized = gaf_resized[:img_size, :img_size]
        
        return gaf_resized


class MarkovTransitionField:
    """Convert time series to Markov Transition Field (MTF)"""
    
    @staticmethod
    def encode(ts: np.ndarray, img_size: int = Config.IMG_SIZE, 
               n_bins: int = Config.N_BINS) -> np.ndarray:
        """Transform time series to MTF"""
        # Ensure input is numpy array
        ts = np.asarray(ts, dtype=np.float64).flatten()
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_norm = scaler.fit_transform(ts.reshape(-1, 1)).flatten()
        
        bins = np.linspace(0, 1, n_bins + 1)
        ts_quantized = np.digitize(ts_norm, bins) - 1
        # Clip to ensure indices are within bounds [0, n_bins-1]
        ts_quantized = np.clip(ts_quantized, 0, n_bins - 1)
        
        mtf = np.zeros((n_bins, n_bins))
        for i in range(len(ts_quantized) - 1):
            mtf[ts_quantized[i], ts_quantized[i + 1]] += 1
        
        mtf = mtf / (mtf.sum() + 1e-8)
        
        # Resize using scipy
        zoom_factor = img_size / n_bins
        mtf_resized = ndimage.zoom(mtf, zoom_factor, order=1)
        mtf_resized = mtf_resized[:img_size, :img_size]
        
        return mtf_resized


class RecurrencePlot:
    """Convert time series to Recurrence Plot (RP)"""
    
    @staticmethod
    def encode(ts: np.ndarray, img_size: int = Config.IMG_SIZE, 
               threshold: float = Config.RECURRENCE_THRESHOLD) -> np.ndarray:
        """Transform time series to RP"""
        # Ensure input is numpy array
        ts = np.asarray(ts, dtype=np.float64).flatten()
        
        scaler = MinMaxScaler()
        ts_norm = scaler.fit_transform(ts.reshape(-1, 1)).flatten()
        
        n = len(ts_norm)
        rp = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                rp[i, j] = 1 if abs(ts_norm[i] - ts_norm[j]) < threshold else 0
        
        # Resize using scipy
        zoom_factor = img_size / n
        rp_resized = ndimage.zoom(rp, zoom_factor, order=1)
        rp_resized = rp_resized[:img_size, :img_size]
        
        return rp_resized


class TimeSeriesImageEncoder:
    """Orchestrates encoding of time series into multiple image representations"""
    
    def __init__(self, img_size: int = Config.IMG_SIZE):
        self.img_size = img_size
        self.gaf_encoder = GramianAngularField()
        self.mtf_encoder = MarkovTransitionField()
        self.rp_encoder = RecurrencePlot()
    
    def encode(self, ts: np.ndarray) -> tuple:
        """Encode time series into all three representations"""
        gaf = self.gaf_encoder.encode(ts, self.img_size)
        mtf = self.mtf_encoder.encode(ts, self.img_size)
        rp = self.rp_encoder.encode(ts, self.img_size)
        
        return gaf, mtf, rp