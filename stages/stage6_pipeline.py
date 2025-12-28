'''
Created on 11 Dec 2025

@author: Dr. Mike
Stage 6: Complete IFFA-TSCI-CNN Pipeline Orchestration

Orchestrates all stages of the IFFA-TSCI-CNN methodology into a unified pipeline.
Chains Stage 1-5 together to create an end-to-end automated ARIMA model selection system.

Pipeline Flow:
1. Data Preparation:
   - Takes list of time series
   - Automatically finds optimal ARIMA(p,d,q) for each (Stage 3)
   - Creates training labels
   
2. Image Encoding (Stage 1):
   - Encodes each time series to GAF, MTF, RP images
   
3. Feature Fusion (Stage 2):
   - Applies IFFA to combine and augment images
   - Creates 64x64x3 images
   
4. CNN Training (Stage 4):
   - Trains ResNet50 to predict ARIMA orders
   - Uses Stage 3 optimal orders as labels
   
5. Validation (Stage 5):
   - Evaluates model with chosen validation method
   - Reports accuracy metrics
   
6. Final Training:
   - Trains final model on all data for production use

Inference (predict):
- Takes new time series
- Encodes through stages 1-2
- CNN predicts best ARIMA(p,d,q)
- Returns prediction and confidence score

Main Methods:
- prepare_data(): Prepare and encode all training data
- train_and_validate(): Train and validate CNN
- predict(): Make prediction on new time series
'''
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from config import Config
from .stage1_image_encoding import TimeSeriesImageEncoder

#weighted fusion
#from .stage2_feature_fusion import InformationFusionFeatureAugmentation
#------------------------------------------
#advanced fusion
from .stage2_feature_optimal_fusion import OptimalInformationFusion
#------------------------------------------------------------
from .stage3_arima_evaluation import ARIMAParameterOptimizer
#from .stage3_arima_evaluation_aic import ARIMAParameterOptimizerAIC as ARIMAParameterOptimizer

from .stage4_cnn_classifier import CNNClassifier
from .stage5_validation import TransferLearningValidation, StratifiedKFoldValidation

class IFFATSCICNNPipeline:
    """Complete IFFA-TSCI-CNN pipeline orchestrating all stages"""
    
    def __init__(self, img_size: int = Config.IMG_SIZE, 
                 batch_size: int = Config.BATCH_SIZE):
        self.img_size = img_size
        self.batch_size = batch_size
        self.encoder = None
        self.iffa = None
        self.cnn = None
        self.label_encoder = None
    
    def prepare_data(self, time_series_list: list, arima_orders: list = None) -> tuple:
        """Complete data preparation: encode, fuse, and prepare labels
        
        Args:
            time_series_list: List of time series arrays
            arima_orders: Optional list of optimal ARIMA orders. If None, will be computed automatically
        """
        self.encoder = TimeSeriesImageEncoder(self.img_size)
        #weighted
        #self.iffa = InformationFusionFeatureAugmentation()
        #advanced
        self.iffa = OptimalInformationFusion()
        
        # If arima_orders not provided, find them automatically
        if arima_orders is None:
            print("\n[STAGE 3] Finding optimal ARIMA parameters for each time series...")
            arima_orders = []
            for i, ts in enumerate(time_series_list):
                print(f"  Series {i+1}/{len(time_series_list)}:")
                try:
                    optimizer = ARIMAParameterOptimizer(ts)
                    best_order, mape = optimizer.find_optimal_order()
                    arima_orders.append(best_order)
                except Exception as e:
                    print(f"    Warning: Error finding order, using default (1,1,1): {str(e)}")
                    arima_orders.append((1, 1, 1))
        
        images = []
        order_strings = []
        
        print("\n[STAGE 1] Encoding time series to images...")
        for i, (ts, order) in enumerate(zip(time_series_list, arima_orders)):
            gaf, mtf, rp = self.encoder.encode(ts)
            fused_augmented = self.iffa.process(gaf, mtf, rp)
            images.append(fused_augmented)
            order_str = f"({order[0]},{order[1]},{order[2]})"
            order_strings.append(order_str)
        
        print(f"[STAGE 2] Applying Information Fusion Feature Augmentation (IFFA)...")
        
        X = np.array(images)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(order_strings)
        y = np.eye(len(self.label_encoder.classes_))[y_encoded]
        
        # Create mapping for CNN
        self.cnn = CNNClassifier(self.img_size, self.batch_size)
        for idx, order_str in enumerate(self.label_encoder.classes_):
            order = tuple(map(int, order_str.strip('()').split(',')))
            self.cnn.label_to_order[idx] = order
        
        print(f"  ✓ Data prepared - Shape: {X.shape}, Classes: {len(self.label_encoder.classes_)}")
        print(f"  ✓ Found ARIMA orders: {list(set(arima_orders))}")
        
        return X, y
    
    def train_and_validate(self, X: np.ndarray, y: np.ndarray,
                          validation_method: str = Config.VALIDATION_METHOD) -> dict:
        """Stage 4 & 5: Train CNN and validate"""
        print(f"\n[STAGE 4] Building CNN Classifier (ResNet50)...")
        print(f"[STAGE 5] Training and Validating ({validation_method.upper()})...")
        
        if validation_method.lower() == 'tl':
            validator = TransferLearningValidation(self.cnn)
            results = validator.validate(X, y)
        else:  # skfcv
            validator = StratifiedKFoldValidation(self.cnn)
            results = validator.validate(X, y)
        
        # Train final model on all data for predictions
        print("    Training final model on all data for predictions...")
        self.cnn.train(X, y, epochs=Config.EPOCHS, validation_split=0.0)
        
        return results
    
    def predict(self, time_series: np.ndarray) -> tuple:
        """Predict best ARIMA order for new time series"""
        gaf, mtf, rp = self.encoder.encode(time_series)
        fused_augmented = self.iffa.process(gaf, mtf, rp)
        X_batch = np.expand_dims(fused_augmented, axis=0)
        
        predicted_orders, confidences = self.cnn.predict(X_batch)
        return predicted_orders[0], confidences[0]