This code implements the ideas presented in:'Two forecasting model selection methods based on time series image feature augmentation' Wentao Jiang, Quan Wang1 & Hongbo Li

# IFFA-TSCI-CNN: Time Series Forecasting Model Selection

A deep learning approach for automated ARIMA model parameter selection using Image Encoding, Information Fusion Feature Augmentation, and Convolutional Neural Networks.

## Overview

This project implements the **IFFA-TSCI-CNN methodology** for intelligent ARIMA(p,d,q) parameter selection. Instead of manually testing hundreds of ARIMA configurations, the CNN learns to predict the optimal parameters by recognising patterns in time series when encoded as images.

### Key Innovation

Transform time series forecasting into an **image classification problem**:
- Convert 1D time series → 3 different 2D images
- Fuse images intelligently with learned weights
- Train CNN to predict the best ARIMA parameters
- Forecast using selected parameters

## Project Structure
```
iffa_tsci_cnn_project/
├── main.py                          # Entry point - runs complete pipeline
├── main_forecast.py                 # Interactive forecasting interface
├── config.py                        # Global configuration
├── requirements.txt                 # Dependencies
├── README.md                        # This file
│
├── stages/                          # 7-Stage methodology
│   ├── stage1_image_encoding.py     # Time series → Images (GAF, MTF, RP)
│   ├── stage2_feature_fusion.py     # IFFA: Fuse & Augment features
│   ├── stage2_feature_optimal_fusion.py  # Alternative: Learnable fusion
│   ├── stage3_arima_evaluation.py   # Find optimal ARIMA parameters
│   ├── stage4_cnn_classifier.py     # ResNet50-based CNN model
│   ├── stage5_validation.py         # Transfer Learning & S-FCV validation
│   ├── stage6_pipeline.py           # Orchestrate all stages
│   └── stage7_forecasting.py        # Generate forecasts
│
├── utils/                           # Utilities
│   ├── data_generator.py            # Generate synthetic ARIMA data
│   └── visualizer.py                # Results visualization
│
└── data/
    └── outputs/                     # Generated results
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

1. **Clone/Download the project**
```bash
   cd iffa_tsci_cnn_project
```

2. **Create virtual environment** (recommended)
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

## Quick Start

### 1. Run Full Pipeline
Generates 30 synthetic ARIMA time series, trains CNN, and forecasts:
```bash
python main.py
```

**Output:**
- Data generation logs
- ARIMA parameter search results
- CNN training with 5-fold validation
- Forecast with confidence intervals
- Model metrics (AIC, BIC)

### 2. Interactive Forecasting
Forecast your own time series or load from files:
```bash
python main_forecast.py
```

**Menu options:**
1. Train model on synthetic data
2. Load time series from CSV file
3. Enter time series manually
4. Exit

### 3. Load Time Series from File

Create a CSV file (e.g., `my_data.csv`):
```
100
105
110
108
115
120
...
```

Then in interactive mode, choose option 2 and provide the file path.

### 4. Manual Time Series Input

In interactive mode, choose option 3 and enter values:
```
100 105 110 108 115 120 125 130 128 135
```

## Configuration

Edit `config.py` to customise:
```python
# Image encoding
IMG_SIZE = 64              # Resolution of encoded images
N_BINS = 20               # Quantisation bins for MTF

# ARIMA optimisation
MAX_P = 5                 # Max autoregressive order
MAX_D = 3                 # Max differencing order
MAX_Q = 5                 # Max moving average order

# CNN training
BATCH_SIZE = 32           # Training batch size
EPOCHS = 30               # Training epochs
LEARNING_RATE = 0.001     # Adam optimiser learning rate

# Validation
VALIDATION_METHOD = 'skfcv'  # 'tl' or 'skfcv'
N_SPLITS = 5              # For cross-validation

# Data generation
N_SAMPLES = 30            # Synthetic time series to generate
TS_LENGTH = 150           # Length of each time series
RANDOM_SEED = 42          # Reproducibility
```

## Methodology

### Stage 1: Time Series Image Encoding

Converts 1D time series into 3 complementary 2D image representations:

**Gramian Angular Field (GAF)**
- Converts values to polar coordinates
- Creates an angular cosine distance matrix
- Captures magnitude relationships

**Markov Transition Field (MTF)**
- Quantises values into discrete states
- Records transition probabilities
- Captures temporal dynamics

**Recurrence Plot (RP)**
- Calculates distance matrix
- Binary representation of recurrence
- Shows dynamical behaviour

### Stage 2: Information Fusion Feature Augmentation (IFFA)

Intelligently combines 3 images:

**Basic Fusion:**
- Weighted average: GAF(0.4) + MTF(0.35) + RP(0.25)
- Creates a single fused 64×64 image

**Feature Augmentation:**
- Channel 1: Original fused image
- Channel 2: Smoothed version
- Channel 3: Edge-detected version
- Output: 64×64×3 augmented image

**Optimal Fusion (Alternative):**
- Extract edge, texture, and frequency features
- Learnable fusion weights (trained with CNN)
- Multiple fusion strategies
- More sophisticated representation

### Stage 3: ARIMA Parameter Optimisation

Grid search for optimal (p,d,q):

**Criterion Options:**
- **MAPE**: Minimise forecast error on test set
- **AIC**: Minimise information criterion (penalises complexity)

Tests all combinations:
- p ∈ [0, 1, 2, 3]
- d ∈ [0, 1, 2]
- q ∈ [0, 1, 2, 3]
- Total: 48 models tested per series

### Stage 4: CNN Classification

ResNet50-based classifier:
- **Input**: 64×64×3 images
- **Architecture**: Pre-trained ResNet50 + custom head
- **Output**: Probability distribution over ARIMA orders
- **Training**: Labels from Stage 3 optimisation

### Stage 5: Validation

Two validation schemes:

**Transfer Learning (TL)**
- Simple 80/20 train-test split
- Fast, suitable for prototyping

**Stratified K-Fold Cross Validation (S-FCV)**
- Multiple folds with class balance
- Handles imbalanced data
- More robust performance estimates

### Stage 6: Pipeline Orchestration

Chains all stages:
```
Generate → Encode → Fuse → Find Orders → Train CNN → Validate
30 Series   Images   IFFA   Stage 3       Stage 4    Stage 5
```

### Stage 7: Forecasting

For new time series:
1. Encode into images
2. Apply IFFA
3. CNN predicts best ARIMA(p,d,q)
4. Fit ARIMA with predicted parameters
5. Generate 12-step forecast
6. Return forecasts with confidence intervals and metrics

## Output Examples

### Training Output
```
[STAGE 3] Finding optimal ARIMA parameters for each time series...
  Series 1/30:
  Grid searching ARIMA parameters (p=3, d=2, q=3)...
    Optimal order: ARIMA(1, 1, 3) with MAPE: 115.5761

[STAGE 4] Building CNN Classifier (ResNet50)...
[STAGE 5] Training and Validating (SKFCV)...
    Performing Stratified 5-Fold Cross Validation...
      Fold 1/5... Accuracy: 0.8500
      Fold 2/5... Accuracy: 0.9000
      Fold 3/5... Accuracy: 0.8750
      Fold 4/5... Accuracy: 0.9250
      Fold 5/5... Accuracy: 0.9500

[RESULTS SUMMARY]
Mean Accuracy: 0.9000 (±0.0327)
```

### Forecasting Output
```
[FORECASTING] Test Time Series
==================================================

[ADF Test] Test Time Series
Test Statistic: -2.1234
p-value: 0.2345
✗ Series is non-stationary (p-value > 0.05)

[Model Fitting] ARIMA(1, 1, 1)
✓ Model fitted successfully

[Model Summary] ARIMA(1, 1, 1)
[Forecasting] Generating a 12-step ahead forecast
✓ Forecast generated for 12 steps

[Forecast Values] ARIMA(1, 1, 1)
            Forecast
2024-02-01    12.3456
2024-03-01    12.4567
2024-04-01    12.5678
...

[Model Metrics]
  AIC: 245.1234
  BIC: 251.5678
  LogLikelihood: -120.5617
```

## Performance Metrics

### Training Metrics
- **Accuracy**: % of correctly predicted ARIMA orders
- **Loss**: Cross-entropy loss

### Forecasting Metrics
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion
- **Confidence Intervals**: 95% bounds on forecasts

## Key Features

✅ **Automated Parameter Selection**: No manual ARIMA tuning
✅ **Multi-Image Encoding**: GAF + MTF + RP for rich representation
✅ **Intelligent Fusion**: IFFA combines complementary information
✅ **Deep Learning**: ResNet50 learns complex patterns
✅ **Robust Validation**: Both TL and S-FCV schemes
✅ **Flexible Input**: File, manual, or synthetic data
✅ **Production Ready**: Complete forecasting pipeline
✅ **Configurable**: Easy to adjust all parameters

## Experimental Features

### Alternative Fusion Methods

Use optimal learnable fusion instead of fixed weights:

In `stage6_pipeline.py`, change:
```python
# FROM:
from .stage2_feature_fusion import InformationFusionFeatureAugmentation
self.iffa = InformationFusionFeatureAugmentation()

# TO:
from .stage2_feature_optimal_fusion import OptimalInformationFusion
self.iffa = OptimalInformationFusion()
```

### Alternative ARIMA Optimisation

Use AIC instead of MAPE:

In `stage6_pipeline.py`, change:
```python
# FROM:
from .stage3_arima_evaluation import ARIMAParameterOptimizer

# TO:
from .stage3_arima_evaluation_aic import ARIMAParameterOptimizerAIC as ARIMAParameterOptimizer
```

## Troubleshooting

### Out of Memory
Reduce `BATCH_SIZE` or `N_SAMPLES` in `config.py`

### Slow Training
- Reduce `EPOCHS` or `N_SAMPLES`
- Enable GPU: Verify CUDA in `config.py`
- Use `VALIDATION_METHOD = 'tl'` instead of `'skfcv'`

### Poor Forecasts
- Increase `N_SAMPLES` for better CNN training
- Expand ARIMA ranges: `MAX_P`, `MAX_D`, `MAX_Q`
- Use longer time series: `TS_LENGTH > 100`

## Requirements

See `requirements.txt`:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
torchvision>=0.11.0
statsmodels>=0.13.0
matplotlib>=3.4.0
scipy>=1.7.0
tqdm>=4.62.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Architecture Diagram
```
Time Series (1D)
    ↓
┌─────────────────────────────┐
│   Stage 1: Encoding         │
│ GAF | MTF | RP (64×64 each) │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Stage 2: IFFA               │
│ Fuse + Augment (64×64×3)    │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Stage 3: ARIMA Optimisation │
│ Grid Search → Best (p,d,q)  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Stage 4: CNN Training       │
│ ResNet50 classifier         │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Stage 5: Validation         │
│ TL or S-FCV                 │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Stage 6: Pipeline Ready     │
│ Predict & Forecast          │
└──────────────┬──────────────┘
               ↓
        Forecast Results
    (values + confidence)
```

## References

The IFFA-TSCI-CNN methodology combines:
- **Time Series Imaging**: GAF, MTF, RP encoding techniques
- **Feature Fusion**: Information fusion for augmentation
- **Deep Learning**: ResNet50 for pattern recognition
- **ARIMA Forecasting**: Statistical model selection

## License

This project is provided for educational and research purposes.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

## Contact & Support

For issues or questions:
1. Check the Troubleshooting section
2. Review configuration in `config.py`
3. Examine output logs for specific errors

## Citation

If you use this project, please cite:
```
IFFA-TSCI-CNN: Time Series Forecasting Model Selection
Using Image Encoding and Deep Learning - Michael Leznik
2025
```

## Changelog

### v1.0 (Current)
- Complete 7-stage pipeline
- GAF, MTF, RP encoding
- Basic and optimal IFFA fusion
- MAPE and AIC optimisation
- ResNet50 CNN classifier
- Transfer Learning and S-FCV validation
- Interactive forecasting interface
- Comprehensive documentation

---

**Happy Forecasting! 🚀**
