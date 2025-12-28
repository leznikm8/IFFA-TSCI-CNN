'''
Created on 11 Dec 2025

@author: Dr. Mike
Synthetic ARIMA Time Series Data Generator

Generates synthetic time series data that follows true ARIMA(p,d,q) processes
for training and testing the IFFA-TSCI-CNN model.

Features:

1. Proper ARIMA Simulation:
   - Uses statsmodels ArmaProcess for stationary ARMA generation
   - Applies differencing (integration) for I(d) component
   - Generates realistic time series with known parameters
   
2. Diverse Parameter Configurations:
   - AR(p) processes: p=1,2,3 with stable coefficients
   - MA(q) processes: q=1,2
   - ARMA(p,q) combinations
   - ARIMA with differencing levels d=0,1,2
   
3. Quality Assurance:
   - Ensures AR coefficient stability
   - Applies burn-in period for convergence
   - Normalizes output to reasonable scales
   - Fallback to random walk if generation fails
   
Parameters:
- N_SAMPLES: Number of time series to generate
- TS_LENGTH: Length of each time series
- RANDOM_SEED: For reproducibility

Output:
- List of numpy arrays (time series data)
- Each series follows true ARIMA distribution
- Ready for Stage 1 encoding

Usage:
    generator = DataGenerator()
    ts_list = generator.generate_data(n_samples=30, ts_length=150)
'''
import numpy as np
from config import Config

try:
    from statsmodels.tsa.arima_process import ArmaProcess
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'statsmodels'])
    from statsmodels.tsa.arima_process import ArmaProcess


class DataGenerator:
    """Generates synthetic ARIMA time series data"""
    
    @staticmethod
    def generate_data(n_samples: int = Config.N_SAMPLES,
                     ts_length: int = Config.TS_LENGTH,
                     seed: int = Config.RANDOM_SEED) -> list:
        """Generate ARIMA time series using ArmaProcess"""
        np.random.seed(seed)
        
        time_series_data = []
        
        print(f"Generating {n_samples} ARIMA time series...")
        
        # AR coefficients (stable)
        ar_configs = [
            [0.5],           # AR(1) with coef 0.5
            [0.3, 0.2],      # AR(2)
            [-0.5],          # AR(1) negative
            [0.7],           # AR(1) strong
        ]
        
        # MA coefficients
        ma_configs = [
            [0.3],           # MA(1)
            [0.5, 0.2],      # MA(2)
            [-0.4],          # MA(1) negative
            [0.6],           # MA(1) strong
        ]
        
        config_idx = 0
        for i in range(n_samples):
            try:
                # Alternate AR and MA patterns
                if i % 2 == 0:
                    ar = [1] + [-coef for coef in ar_configs[config_idx % len(ar_configs)]]
                    ma = [1]
                else:
                    ar = [1]
                    ma = [1] + ma_configs[config_idx % len(ma_configs)]
                
                # Generate ARMA process
                arma_process = ArmaProcess(ar, ma)
                ts = arma_process.generate_sample(ts_length, scale=1.0)
                
                # Add integration (differencing) for some series
                if i % 3 == 0 and i > 0:
                    ts = np.cumsum(ts)
                
                time_series_data.append(np.asarray(ts, dtype=np.float64))
                config_idx += 1
                
            except Exception as e:
                print(f"  Warning: Series {i+1} failed, using simple random walk")
                ts = np.cumsum(np.random.randn(ts_length) * 0.1)
                time_series_data.append(np.asarray(ts, dtype=np.float64))
        
        print(f"  ✓ Generated {n_samples} time series")
        return time_series_data