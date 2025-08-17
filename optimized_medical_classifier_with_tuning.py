import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, log_loss
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
import random
from torch.amp import autocast, GradScaler
import math
import gc
from collections import Counter, defaultdict
import copy
import warnings
from scipy.special import softmax
import optuna
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(42)

# Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate metrics with focus on target values"""
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Calculate confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Basic metrics
    sensitivity = recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Class-specific F1 scores
    ce_mask = y_true == 0
    laa_mask = y_true == 1
    
    # CE class
    true_ce = y_true[ce_mask]
    pred_ce = y_pred[ce_mask]
    ce_f1 = f1_score(true_ce, pred_ce, zero_division=0) if len(true_ce) > 0 else 0
    
    # LAA class
    true_laa = y_true[laa_mask]
    pred_laa = y_pred[laa_mask]
    laa_f1 = f1_score(true_laa, pred_laa, zero_division=0) if len(true_laa) > 0 else 0
    
    # Balanced F1
    balanced_f1 = (ce_f1 * laa_f1) ** 0.5 if ce_f1 > 0 and laa_f1 > 0 else 0
    
    # AUC-ROC (target > 80%)
    auc_roc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    
    # PR-AUC (target > 90%)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Log loss (target < 0.64)
    eps = 1e-15
    y_probs_clipped = np.clip(y_probs, eps, 1 - eps)
    wcll = log_loss(y_true, y_probs_clipped)
    
    # G-mean (geometric mean of sensitivity and specificity)
    g_mean = (sensitivity * specificity) ** 0.5 if sensitivity > 0 and specificity > 0 else 0
    
    # MCC - Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Bundle metrics
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,  # Target > 90%
        'specificity': specificity,  # Target > 90%
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_weighted': f1_weighted,  # Target > 70%
        'ce_f1': ce_f1,
        'laa_f1': laa_f1,
        'balanced_f1': balanced_f1,
        'mcc': mcc,
        'auc_roc': auc_roc,  # Target > 80%
        'pr_auc': pr_auc,     # Target > 90% 
        'wcll': wcll,   # Target < 0.64
        'g_mean': g_mean,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }
    
    # Add target evaluation status
    metrics['targets_met'] = {
        'sensitivity': sensitivity >= 0.90,
        'specificity': specificity >= 0.90,
        'f1_weighted': f1_weighted >= 0.70,
        'wcll': wcll <= 0.64,
        'pr_auc': pr_auc >= 0.90,
        'auc_roc': auc_roc >= 0.80
    }
    
    # Overall target achievement
    metrics['target_achievement'] = sum(metrics['targets_met'].values()) / len(metrics['targets_met'])
    
    return metrics

# Enhanced configuration with hyperparameter tuning support
class TunableConfig:
    # Data parameters - Fixed
    FEATURE_DIM = 1024
    NUM_TILES = 16
    
    # Tunable training parameters
    def __init__(self, trial=None):
        if trial is not None:
            # Hyperparameter tuning mode
            self.BATCH_SIZE = trial.suggest_categorical('batch_size', [8, 16, 32])
            self.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
            self.WEIGHT_DECAY = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
            self.NUM_EPOCHS = trial.suggest_int('num_epochs', 20, 50)
            self.PATIENCE = max(10, self.NUM_EPOCHS // 3)
            
            # Model architecture parameters
            self.HIDDEN_DIM = trial.suggest_categorical('hidden_dim', [128, 256, 512])
            self.DROPOUT_RATE = trial.suggest_float('dropout_rate', 0.1, 0.5)
            self.FEATURE_DROPOUT = trial.suggest_float('feature_dropout', 0.05, 0.3)
            self.NUM_ATTENTION_HEADS = trial.suggest_categorical('num_attention_heads', [2, 4, 8])
            self.NUM_LAYERS = trial.suggest_int('num_layers', 1, 4)
            
            # Class weighting
            self.CE_CLASS_WEIGHT = trial.suggest_float('ce_class_weight', 0.8, 2.0)
            self.LAA_CLASS_WEIGHT = trial.suggest_float('laa_class_weight', 0.8, 2.0)
            
            # Advanced loss settings
            self.USE_FOCAL_LOSS = trial.suggest_categorical('use_focal_loss', [True, False])
            if self.USE_FOCAL_LOSS:
                self.FOCAL_GAMMA = trial.suggest_float('focal_gamma', 1.0, 3.0)
                self.FOCAL_ALPHA = trial.suggest_float('focal_alpha', 0.2, 0.8)
            else:
                self.FOCAL_GAMMA = 2.0
                self.FOCAL_ALPHA = 0.4
                
            self.USE_LABEL_SMOOTHING = trial.suggest_categorical('use_label_smoothing', [True, False])
            if self.USE_LABEL_SMOOTHING:
                self.LABEL_SMOOTHING = trial.suggest_float('label_smoothing', 0.05, 0.2)
            else:
                self.LABEL_SMOOTHING = 0.0
                
            # Advanced training settings
            self.USE_SWA = trial.suggest_categorical('use_swa', [True, False])
            if self.USE_SWA:
                self.SWA_START = trial.suggest_float('swa_start', 0.3, 0.7)
                self.SWA_LR = trial.suggest_float('swa_lr', 1e-6, 1e-4, log=True)
            else:
                self.SWA_START = 0.5
                self.SWA_LR = 5e-6
                
            self.USE_EMA = trial.suggest_categorical('use_ema', [True, False])
            if self.USE_EMA:
                self.EMA_DECAY = trial.suggest_float('ema_decay', 0.99, 0.9999)
            else:
                self.EMA_DECAY = 0.999
                
            self.GRADIENT_ACCUMULATION = trial.suggest_categorical('gradient_accumulation', [2, 4, 8])
            
            # LR Scheduler settings
            self.USE_ONECYCLE_LR = trial.suggest_categorical('use_onecycle_lr', [True, False])
            if self.USE_ONECYCLE_LR:
                self.PCT_START = trial.suggest_float('pct_start', 0.1, 0.5)
            else:
                self.PCT_START = 0.3
                
            # Multi-sample dropout
            self.USE_MULTI_SAMPLE_DROPOUT = trial.suggest_categorical('use_multi_sample_dropout', [True, False])
            if self.USE_MULTI_SAMPLE_DROPOUT:
                self.MULTI_SAMPLE_DROPOUT_COUNT = trial.suggest_int('multi_sample_dropout_count', 2, 6)
            else:
                self.MULTI_SAMPLE_DROPOUT_COUNT = 4
                
        else:
            # Default optimized parameters based on analysis
            self.BATCH_SIZE = 16  # Increased from 8
            self.LEARNING_RATE = 5e-5  # Increased from 1e-5
            self.WEIGHT_DECAY = 0.01  # Reduced from 0.015
            self.NUM_EPOCHS = 35
            self.PATIENCE = 12
            
            # Model parameters - slightly reduced complexity
            self.HIDDEN_DIM = 256
            self.DROPOUT_RATE = 0.25  # Reduced from 0.3
            self.FEATURE_DROPOUT = 0.1  # Reduced from 0.15
            self.NUM_ATTENTION_HEADS = 4
            self.NUM_LAYERS = 2
            
            # Class weighting - better balanced
            self.CE_CLASS_WEIGHT = 1.3
            self.LAA_CLASS_WEIGHT = 1.2
            
            # Advanced loss settings
            self.USE_FOCAL_LOSS = True
            self.FOCAL_GAMMA = 2.0
            self.FOCAL_ALPHA = 0.4
            self.USE_LABEL_SMOOTHING = True
            self.LABEL_SMOOTHING = 0.08  # Reduced from 0.1
            
            # Advanced training settings
            self.USE_SWA = True
            self.SWA_START = 0.5
            self.SWA_LR = 8e-6  # Increased from 5e-6
            self.USE_EMA = True
            self.EMA_DECAY = 0.999
            self.GRADIENT_ACCUMULATION = 4
            
            # LR Scheduler settings
            self.USE_ONECYCLE_LR = True
            self.PCT_START = 0.3
            
            # Multi-sample dropout
            self.USE_MULTI_SAMPLE_DROPOUT = True
            self.MULTI_SAMPLE_DROPOUT_COUNT = 3  # Reduced from 4
        
        # Fixed parameters
        self.OUTPUT_DIM = 2
        self.OPTIMAL_THRESHOLD_SEARCH = True
        self.USE_TEMPERATURE_SCALING = True
        self.INITIAL_TEMPERATURE = 1.0
        self.USE_GRADIENT_CLIPPING = True
        self.GRADIENT_CLIP_VALUE = 1.0
        self.CHECK_OVERFITTING = True
        self.OVERFITTING_THRESHOLD = 0.2  # Reduced from 0.25
        self.PREVENT_VAL_DEGRADATION = True
        self.ENSURE_BALANCED_PREDS = True
        self.MIN_CLASS_PERCENT = 0.35  # More restrictive
        self.FIXED_THRESHOLD = 0.45
        
        # Paths
        self.SAVE_DIR = "./tuned_medical_models"
        self.FEATURES_DIR = "../input/train-attention-fusion-features-16-tiles"

# Feature Transformer with improved caching
class FeatureTransformer:
    def __init__(self, method='power'):
        self.method = method
        if method == 'power':
            self.transformer = PowerTransformer(method='yeo-johnson')
        elif method == 'robust':
            self.transformer = RobustScaler(quantile_range=(5, 95))  # Wider range
        else:
            self.transformer = StandardScaler()
        self.fitted = False
        
    def fit(self, features):
        """Fit the transformer on a batch of features"""
        if features.shape[0] > 0:
            # Reshape to 2D if needed
            if len(features.shape) > 2:
                flat_features = features.reshape(-1, features.shape[-1])
            else:
                flat_features = features
                
            self.transformer.fit(flat_features)
            self.fitted = True
        
    def transform(self, features):
        """Transform features and maintain original shape"""
        if not self.fitted:
            return features
            
        # Get original shape and reshape to 2D
        original_shape = features.shape
        if len(original_shape) > 2:
            flat_features = features.reshape(-1, original_shape[-1])
        else:
            flat_features = features
            
        # Transform features
        transformed = self.transformer.transform(flat_features)
        
        # Reshape back to original shape
        if len(original_shape) > 2:
            transformed = transformed.reshape(original_shape)
            
        return transformed

# Improved medical feature augmenter with more sophisticated techniques
class MedicalFeatureAugmenter:
    def __init__(self, config):
        self.config = config
        self.noise_level = 0.03  # Reduced noise
        self.mixup_alpha = 0.2  # For mixup augmentation
        
    def __call__(self, features, label=None):
        features = features.clone()
        
        # Apply augmentation with probability
        if random.random() < 0.6:  # Increased probability
            # 1. Feature dropout
            if random.random() < 0.5:
                mask = torch.rand_like(features) > self.config.FEATURE_DROPOUT
                features = features * mask
            
            # 2. Gaussian noise
            if random.random() < 0.4:
                noise = torch.randn_like(features) * self.noise_level
                features = features + noise
            
            # 3. Feature scaling (simulate different imaging conditions)
            if random.random() < 0.3:
                scale_factor = torch.normal(1.0, 0.1, size=(features.shape[0], 1))
                scale_factor = torch.clamp(scale_factor, 0.8, 1.2)
                features = features * scale_factor
                
        return features

# Optimized dataset with better error handling
class OptimizedDataset(Dataset):
    def __init__(self, df, data_dir, augmenter=None, use_cache=True, training=True, transformer=None):
        self.df = df
        self.data_dir = data_dir
        self.augmenter = augmenter
        self.cache = {} if use_cache else None
        self.training = training
        self.transformer = transformer
        self.error_count = 0
        
        # Map patient_ids
        if 'patient_id' in self.df.columns:
            self.patient_ids = self.df['patient_id'].unique()
            self.patient_id_map = {pid: i for i, pid in enumerate(self.patient_ids)}
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        image_id = self.df.iloc[index].image_id
        
        # Handle label extraction
        if 'label' in self.df.columns:
            label_value = self.df.iloc[index].label
            # Handle both string and int labels
            if isinstance(label_value, str):
                label = 0 if label_value == "CE" else 1
            else:
                label = label_value
        else:
            # Test data
            label = -1
            
        # Get patient_id if available
        patient_id = -1
        if 'patient_id' in self.df.columns:
            pid = self.df.iloc[index].patient_id
            patient_id = self.patient_id_map.get(pid, -1)
    
        # Try cache first
        if self.cache is not None and image_id in self.cache:
            features = self.cache[image_id]
        else:
            try:
                full_path = f"{self.data_dir}/{image_id}.h5"
                with h5py.File(full_path, 'r') as hdf5_file:
                    tiles = []
                    for i in range(TunableConfig.NUM_TILES):
                        if str(i) in hdf5_file:
                            # Get feature
                            feat = torch.tensor(hdf5_file[str(i)][:])
                            # Remove extra dimensions
                            if len(feat.shape) > 1 and feat.shape[0] == 1:
                                feat = feat.squeeze(0)
                            tiles.append(feat)
                        else:
                            # Consistent random noise for missing tiles
                            seed_val = hash(f"{image_id}_{i}") % 10000
                            torch.manual_seed(seed_val)
                            tiles.append(torch.zeros(TunableConfig.FEATURE_DIM) + 
                                       torch.randn(TunableConfig.FEATURE_DIM) * 0.01)
                    
                    # Stack tiles
                    features = torch.stack(tiles, dim=0)
                    
                    # Apply feature transformation if provided
                    if self.transformer is not None and hasattr(self.transformer, 'transform'):
                        features_np = features.numpy()
                        features_np = self.transformer.transform(features_np)
                        features = torch.tensor(features_np, dtype=torch.float32)
                    
                    # Store in cache
                    if self.cache is not None:
                        self.cache[image_id] = features
                        
            except Exception as e:
                self.error_count += 1
                if self.error_count <= 5:  # Only print first 5 errors
                    print(f"Error loading features for {image_id}: {str(e)}")
                # Use consistent random noise for missing images
                torch.manual_seed(hash(image_id) % 10000)
                features = torch.zeros((TunableConfig.NUM_TILES, TunableConfig.FEATURE_DIM)) + \
                          torch.randn((TunableConfig.NUM_TILES, TunableConfig.FEATURE_DIM)) * 0.01
        
        # Apply augmentation during training
        if self.augmenter is not None and self.training and label != -1:
            features = self.augmenter(features, label)
            
        return features, label, image_id, patient_id

# Create and fit feature transformer with better sampling
def create_feature_transformer(train_df, data_dir, method='power', sample_size=1000):
    """Create and fit a feature transformer on training data"""
    print(f"Creating and fitting {method} feature transformer...")
    
    transformer = FeatureTransformer(method=method)
    
    # Collect features for fitting
    all_features = []
    
    # Stratified sampling to ensure both classes are represented
    ce_samples = train_df[train_df['label'] == 0].sample(n=min(sample_size//2, len(train_df[train_df['label'] == 0])), random_state=42)
    laa_samples = train_df[train_df['label'] == 1].sample(n=min(sample_size//2, len(train_df[train_df['label'] == 1])), random_state=42)
    sample_df = pd.concat([ce_samples, laa_samples])
    
    for idx in tqdm(range(len(sample_df)), desc="Collecting features for transformer"):
        image_id = sample_df.iloc[idx].image_id
        
        try:
            full_path = f"{data_dir}/{image_id}.h5"
            with h5py.File(full_path, 'r') as hdf5_file:
                for i in range(TunableConfig.NUM_TILES):
                    if str(i) in hdf5_file:
                        feat = np.array(hdf5_file[str(i)][:])
                        if len(feat.shape) > 1 and feat.shape[0] == 1:
                            feat = feat.squeeze(0)
                        all_features.append(feat)
        except Exception as e:
            if len(all_features) < 10:  # Only print first few errors
                print(f"Error loading features for transformer: {str(e)}")
    
    # Fit transformer
    if all_features:
        features_array = np.vstack(all_features)
        transformer.fit(features_array)
        print(f"Transformer fitted on {len(features_array)} feature vectors")
    else:
        print("Warning: No features found for transformer fitting")
    
    return transformer

# Improved model architecture with better regularization
class OptimizedModel(nn.Module):
    def __init__(self, config):
        super(OptimizedModel, self).__init__()
        
        self.config = config
        input_dim = config.FEATURE_DIM
        hidden_dim = config.HIDDEN_DIM
        output_dim = config.OUTPUT_DIM
        dropout_rate = config.DROPOUT_RATE
        num_heads = config.NUM_ATTENTION_HEADS
        num_layers = config.NUM_LAYERS
        
        # Initial embedding with improved normalization
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Changed from BatchNorm1d
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Learnable position embeddings
        self.position_embedding = nn.Parameter(torch.zeros(1, config.NUM_TILES, hidden_dim))
        nn.init.normal_(self.position_embedding, std=0.02)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Global pooling with attention
        self.global_pool = GlobalAttentionPool(hidden_dim, dropout_rate)
        
        # Multi-sample dropout for robust feature learning
        if config.USE_MULTI_SAMPLE_DROPOUT:
            self.multi_dropouts = nn.ModuleList([
                nn.Dropout(0.1 + 0.05 * i)  # Reduced dropout range
                for i in range(config.MULTI_SAMPLE_DROPOUT_COUNT)
            ])
            self.classifiers = nn.ModuleList([
                nn.Linear(hidden_dim, output_dim) 
                for _ in range(config.MULTI_SAMPLE_DROPOUT_COUNT)
            ])
        else:
            # Standard classifier with improved architecture
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate * 0.5),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate * 0.3),
                nn.Linear(hidden_dim // 2, output_dim)
            )
        
    def _apply_embeddings(self, x):
        """Apply embeddings efficiently"""
        batch_size, num_tiles, feat_dim = x.shape
        
        # Reshape for processing
        x_flat = x.reshape(-1, feat_dim)
        
        # Apply embedding (LayerNorm handles batching better than BatchNorm)
        embedded = self.embedding(x_flat)
        
        # Reshape back
        embedded = embedded.reshape(batch_size, num_tiles, -1)
        
        # Add position embeddings
        embedded = embedded + self.position_embedding[:, :num_tiles, :]
        
        return embedded
            
    def forward(self, x):
        # Apply initial embedding with position information
        x = self._apply_embeddings(x)
        
        # Apply transformer encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        # Global attention pooling
        pooled = self.global_pool(x)
        
        # Apply multi-sample dropout if enabled
        if hasattr(self, 'multi_dropouts'):
            logits = []
            for i, dropout in enumerate(self.multi_dropouts):
                dropout_features = dropout(pooled)
                logits.append(self.classifiers[i](dropout_features))
            logits = torch.mean(torch.stack(logits), dim=0)
            return logits
        else:
            return self.classifier(pooled)

# Improved transformer encoder block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(TransformerEncoderBlock, self).__init__()
        
        # Pre-layer normalization architecture
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout_rate * 0.3,  # Reduced attention dropout
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),  # Increased expansion ratio
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        
    def forward(self, x):
        # Self-attention with residual connection
        normalized = self.norm1(x)
        attention_output, _ = self.self_attention(normalized, normalized, normalized)
        x = x + self.dropout1(attention_output)
        
        # Feed-forward with residual connection
        normalized = self.norm2(x)
        ff_output = self.feed_forward(normalized)
        x = x + self.dropout2(ff_output)
        
        return x

# Improved global attention pooling
class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(GlobalAttentionPool, self).__init__()
        
        # Multi-head attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout_rate * 0.3)
        
    def forward(self, x):
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        
        # Apply attention pooling
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        
        # Apply dropout
        weighted_sum = self.dropout(weighted_sum)
        
        return weighted_sum

# Improved focal loss
class FocalLossWithSmoothing(nn.Module):
    def __init__(self, config):
        super(FocalLossWithSmoothing, self).__init__()
        self.gamma = config.FOCAL_GAMMA
        self.alpha = config.FOCAL_ALPHA
        self.smoothing = config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0
        self.ce_weight = config.CE_CLASS_WEIGHT
        self.laa_weight = config.LAA_CLASS_WEIGHT
        self.eps = 1e-8
        
    def forward(self, logits, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        
        # Apply label smoothing if enabled
        if self.smoothing > 0:
            smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / 2
            targets_one_hot = smooth_targets
        
        # Calculate focal loss components
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        
        # Alpha weighting
        batch_alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross-entropy loss
        ce_loss = -torch.log(p_t)
        
        # Class weights
        class_weights = torch.ones_like(targets, dtype=torch.float)
        class_weights[targets == 0] = self.ce_weight
        class_weights[targets == 1] = self.laa_weight
        
        # Combined loss
        loss = class_weights * batch_alpha * focal_weight * ce_loss
        
        return loss.mean()

# Enhanced EMA with warmup
class EnhancedEMA:
    def __init__(self, model, decay=0.999, warmup_steps=1000):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.warmup_steps = warmup_steps
        self.step_counter = 0
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        self.step_counter += 1
        
        # Warmup decay adjustment
        if self.step_counter < self.warmup_steps:
            decay = min(self.decay, 1.0 - 1.0 / (self.step_counter + 1))
        else:
            decay = self.decay
            
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = decay * self.shadow[name] + (1.0 - decay) * param.data
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# Temperature scaling for calibration
class TemperatureScaling:
    def __init__(self, temp=1.0):
        self.temperature = temp
        
    def __call__(self, logits):
        return logits / self.temperature
    
    def fit(self, logits, targets):
        from scipy.optimize import minimize
        
        def objective(temp):
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=1)
            probs_class = torch.gather(probs, 1, targets.unsqueeze(1))
            nll = -torch.log(probs_class + 1e-8).mean().item()
            return nll
        
        # Find optimal temperature
        optimal = minimize(objective, x0=self.temperature, 
                          method='L-BFGS-B', bounds=[(0.3, 5.0)])
        
        self.temperature = optimal.x[0]
        return self

# Improved balanced sampler
class ImprovedGeometricMeanSampler:
    def __init__(self, labels, class_weights=None):
        self.labels = np.array(labels)
        self.class_counts = np.bincount(self.labels)
        self.class_weights = class_weights or {i: 1.0 for i in range(len(self.class_counts))}
        
        # Calculate sampling weights using geometric mean
        geometric_mean = np.exp(np.mean(np.log(self.class_counts + 1)))
        sampling_weights = geometric_mean / (self.class_counts + 1)
        
        # Apply class weights with balanced adjustment
        self.weights = np.zeros_like(self.labels, dtype=np.float32)
        
        for i, count in enumerate(self.class_counts):
            mask = self.labels == i
            base_weight = sampling_weights[i] * self.class_weights.get(i, 1.0)
            self.weights[mask] = base_weight
        
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
    def __iter__(self):
        indices = np.random.choice(
            len(self.labels), 
            size=len(self.labels), 
            replace=True, 
            p=self.weights
        )
        return iter(indices.tolist())
        
    def __len__(self):
        return len(self.labels)

# Improved threshold finding with multiple objectives
def find_optimal_threshold(y_true, y_probs, target_sens=0.85, target_spec=0.85):
    """Find optimal threshold with multiple objectives"""
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # More granular threshold search
    thresholds = np.linspace(0.1, 0.9, 81)  # More points
    
    best_score = -float('inf')
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_probs)
        
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        
        # Multi-objective scoring
        g_mean = metrics['g_mean']
        f1_weighted = metrics['f1_weighted']
        
        # Constraint satisfaction
        constraints_met = (sensitivity >= target_sens and specificity >= target_spec)
        
        # Balanced scoring with medical priorities
        balance_penalty = abs(sensitivity - specificity) * 0.1
        medical_score = (sensitivity * 0.4 + specificity * 0.4 + g_mean * 0.2)
        score = medical_score + f1_weighted * 0.3 - balance_penalty
        
        # Bonus for meeting constraints
        if constraints_met:
            score *= 1.1
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics

# Hyperparameter tuning with Optuna
def objective(trial, train_df, val_df):
    """Optuna objective function for hyperparameter tuning"""
    
    # Create configuration with trial parameters
    config = TunableConfig(trial)
    
    try:
        # Train model with current hyperparameters
        model, threshold, temperature, metrics, transformer, history = train_model(
            train_df, val_df, config, fold=0
        )
        
        # Multi-objective optimization
        # Primary: Target achievement (medical requirements)
        target_score = metrics['target_achievement']
        
        # Secondary: G-mean (balanced performance)
        g_mean_score = metrics['g_mean']
        
        # Tertiary: AUC-ROC (overall discrimination)
        auc_score = metrics['auc_roc']
        
        # Combined objective with medical priorities
        combined_score = (target_score * 0.5 + 
                         g_mean_score * 0.3 + 
                         auc_score * 0.2)
        
        # Clean up GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return combined_score
        
    except Exception as e:
        print(f"Trial failed with error: {str(e)}")
        return 0.0

# Improved training function with better monitoring
def train_model(train_df, val_df, config, fold=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create feature transformer
    transformer = create_feature_transformer(train_df, config.FEATURES_DIR, method='power')
    
    # Create augmenter
    augmenter = MedicalFeatureAugmenter(config)
    
    # Create datasets
    train_dataset = OptimizedDataset(
        train_df, 
        config.FEATURES_DIR,
        augmenter=augmenter,
        training=True,
        transformer=transformer
    )
    
    val_dataset = OptimizedDataset(
        val_df, 
        config.FEATURES_DIR,
        training=False,
        transformer=transformer
    )
    
    # Create balanced sampler
    labels = []
    for idx in range(len(train_df)):
        label_value = train_df.iloc[idx].label
        if isinstance(label_value, str):
            label = 0 if label_value == 'CE' else 1
        else:
            label = int(label_value)
        labels.append(label)
    
    class_weights = {
        0: config.CE_CLASS_WEIGHT,
        1: config.LAA_CLASS_WEIGHT
    }
    
    sampler = ImprovedGeometricMeanSampler(labels, class_weights=class_weights)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=0
    )
    
    # Create model
    model = OptimizedModel(config).to(device)
    
    # Loss function
    if config.USE_FOCAL_LOSS:
        criterion = FocalLossWithSmoothing(config)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([config.CE_CLASS_WEIGHT, config.LAA_CLASS_WEIGHT], 
                               device=device)
        )
    
    # Optimizer with improved settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if config.USE_ONECYCLE_LR:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE * 8,  # Reduced multiplier
            total_steps=len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION,
            pct_start=config.PCT_START,
            div_factor=20.0,
            final_div_factor=500.0
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=len(train_loader) * config.NUM_EPOCHS // (4 * config.GRADIENT_ACCUMULATION),
            eta_min=config.LEARNING_RATE / 50
        )
    
    # Initialize advanced training components
    if config.USE_EMA:
        ema = EnhancedEMA(model, decay=config.EMA_DECAY, warmup_steps=1000)
    
    if config.USE_SWA:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer, 
            swa_lr=config.SWA_LR,
            anneal_epochs=3
        )
        swa_start = int(config.NUM_EPOCHS * config.SWA_START)
    
    if config.USE_TEMPERATURE_SCALING:
        temperature_scaler = TemperatureScaling(temp=config.INITIAL_TEMPERATURE)
    
    # Initialize mixed precision
    scaler = GradScaler()
    
    # Training tracking
    best_metrics = {
        'auc_roc': 0.0,
        'f1_weighted': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'wcll': float('inf'),
        'g_mean': 0.0
    }
    
    best_target_achievement = 0.0
    best_epoch = 0
    patience_counter = 0
    
    model_saved = False
    best_model_path = f"{config.SAVE_DIR}/best_model_fold_{fold}.pth"
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc_roc': [], 'val_f1_weighted': [],
        'val_sensitivity': [], 'val_specificity': [],
        'val_wcll': [], 'val_g_mean': [],
        'val_targets_met': [], 'lr': []
    }
    
    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_probs = []
        train_labels = []
        
        optimizer.zero_grad()
        batch_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for batch_idx, (features, labels, _, _) in enumerate(progress_bar):
            features, labels = features.to(device), labels.to(device)
            batch_count += 1
            
            try:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(features)
                    loss = criterion(outputs, labels) / config.GRADIENT_ACCUMULATION
                
                scaler.scale(loss).backward()
                
                # Gradient accumulation and clipping
                if (batch_count % config.GRADIENT_ACCUMULATION == 0) or (batch_idx == len(train_loader) - 1):
                    if config.USE_GRADIENT_CLIPPING:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update schedulers
                    if not config.USE_SWA or epoch < swa_start:
                        scheduler.step()
                    else:
                        swa_scheduler.step()
                
                # Track metrics
                train_loss += loss.item() * config.GRADIENT_ACCUMULATION * features.size(0)
                
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                train_preds.extend(preds.detach().cpu().numpy())
                train_probs.extend(probs[:, 1].detach().cpu().numpy())
                train_labels.extend(labels.detach().cpu().numpy())
                
                # Update EMA
                if config.USE_EMA and ((batch_count % config.GRADIENT_ACCUMULATION == 0) or (batch_idx == len(train_loader) - 1)):
                    ema.update()
                
                progress_bar.set_postfix({'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION:.4f}"})
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        # Update SWA
        if config.USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation phase
        if config.USE_EMA:
            ema.apply_shadow()
            
        val_model = swa_model if (config.USE_SWA and epoch >= swa_start) else model
        val_model.eval()
        
        val_loss = 0.0
        val_preds = []
        val_probs = []
        val_labels = []
        val_logits = []
        
        with torch.no_grad():
            for batch_idx, (features, labels, _, _) in enumerate(val_loader):
                try:
                    features, labels = features.to(device), labels.to(device)
                    
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        outputs = val_model(features)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * features.size(0)
                    val_logits.extend(outputs.detach().cpu())
                    
                    if config.USE_TEMPERATURE_SCALING and epoch > 0:
                        outputs = outputs / temperature_scaler.temperature
                        
                    probs = F.softmax(outputs, dim=1)
                    preds = probs.argmax(dim=1)
                    val_preds.extend(preds.detach().cpu().numpy())
                    val_probs.extend(probs[:, 1].detach().cpu().numpy())
                    val_labels.extend(labels.detach().cpu().numpy())
                
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        if config.USE_EMA:
            ema.restore()
            
        val_loss = val_loss / len(val_loader.dataset)
        
        # Temperature scaling
        if config.USE_TEMPERATURE_SCALING and epoch >= 2 and len(val_logits) > 0:
            val_logits_tensor = torch.stack(val_logits)
            val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
            temperature_scaler.fit(val_logits_tensor, val_labels_tensor)
        
        # Find optimal threshold
        val_labels_np = np.array(val_labels)
        val_probs_np = np.array(val_probs)
        
        threshold, val_metrics = find_optimal_threshold(
            val_labels_np, val_probs_np,
            target_sens=0.85, target_spec=0.85
        )
        
        # Update history
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc_roc'].append(val_metrics["auc_roc"])
        history['val_f1_weighted'].append(val_metrics["f1_weighted"])
        history['val_sensitivity'].append(val_metrics["sensitivity"])
        history['val_specificity'].append(val_metrics["specificity"])
        history['val_wcll'].append(val_metrics["wcll"])
        history['val_g_mean'].append(val_metrics["g_mean"])
        history['val_targets_met'].append(val_metrics["target_achievement"])
        
        # Model saving logic
        should_save = False
        
        # Primary: Target achievement improvement
        target_improvement = val_metrics["target_achievement"] > best_target_achievement * 1.01
        
        # Secondary: G-mean improvement
        g_mean_improvement = val_metrics["g_mean"] > best_metrics["g_mean"] * 1.01
        
        # Combined improvement
        if target_improvement and val_metrics["target_achievement"] >= 0.5:
            should_save = True
        elif g_mean_improvement and val_metrics["g_mean"] >= 0.7:
            should_save = True
        elif not model_saved and epoch >= 5:
            should_save = True
        
        if should_save:
            # Update best metrics
            for key in best_metrics:
                if key == "wcll":
                    best_metrics[key] = min(best_metrics[key], val_metrics[key])
                else:
                    best_metrics[key] = max(best_metrics[key], val_metrics[key])
            
            best_target_achievement = max(best_target_achievement, val_metrics["target_achievement"])
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            save_dict = {
                'epoch': epoch,
                'model_state_dict': val_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'threshold': threshold,
                'temperature': temperature_scaler.temperature if config.USE_TEMPERATURE_SCALING else 1.0,
                'is_swa': (config.USE_SWA and epoch >= swa_start),
                'is_ema': config.USE_EMA,
                'fold': fold,
                'transformer': transformer,
                'config': config
            }
                
            if config.USE_EMA:
                save_dict['ema_shadows'] = ema.shadow
                
            torch.save(save_dict, best_model_path)
            model_saved = True
            
            print(f'✓ New best model saved! Target achievement: {val_metrics["target_achievement"]:.2%}')
            
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.PATIENCE:
            if epoch >= 15 and model_saved:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Load best model
    if model_saved:
        checkpoint = torch.load(best_model_path)
        
        if checkpoint.get('is_swa', False):
            best_model = torch.optim.swa_utils.AveragedModel(OptimizedModel(config))
            best_model.to(device)
            best_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            best_model = OptimizedModel(config).to(device)
            best_model.load_state_dict(checkpoint['model_state_dict'])
            
            if checkpoint.get('is_ema', False) and 'ema_shadows' in checkpoint:
                ema = EnhancedEMA(best_model, decay=config.EMA_DECAY)
                ema.shadow = checkpoint['ema_shadows']
                ema.apply_shadow()
                
        temperature = checkpoint.get('temperature', 1.0)
        threshold = checkpoint.get('threshold', 0.5)
        transformer = checkpoint.get('transformer', None)
        
        return best_model, threshold, temperature, best_metrics, transformer, history
    else:
        return val_model, 0.5, 1.0, val_metrics, transformer, history

# Hyperparameter tuning function
def tune_hyperparameters(train_df, n_trials=50):
    """Run hyperparameter tuning with Optuna"""
    
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Split data for tuning (use a smaller validation set for speed)
    if 'patient_id' in train_df.columns:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(train_df, train_df['label'], groups=train_df['patient_id']))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(train_df, train_df['label']))
    
    tune_train_df = train_df.iloc[train_idx].reset_index(drop=True)
    tune_val_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, tune_train_df, tune_val_df),
        n_trials=n_trials,
        timeout=None,
        show_progress_bar=True
    )
    
    # Results
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results_path = "./tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nTuning results saved to {results_path}")
    
    return study.best_params

# Main function with tuning option
def main(tune_hyperparams=False, n_trials=30):
    seed_everything(42)
    
    # Load data
    train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv")
    test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
    
    print(f"Training with {len(train_df)} samples, testing on {len(test_df)} samples")
    
    # Prepare data
    if 'patient_id' not in train_df.columns:
        train_df['patient_id'] = train_df['image_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    if train_df['label'].dtype == 'object':
        label_map = {'CE': 0, 'LAA': 1}
        train_df['original_label'] = train_df['label']
        train_df['label'] = train_df['label'].map(label_map)
    
    # Hyperparameter tuning
    if tune_hyperparams:
        best_params = tune_hyperparameters(train_df, n_trials=n_trials)
        print("\n" + "="*50)
        print("Training with optimized hyperparameters...")
        print("="*50)
        
        # Create a mock trial object with best parameters
        class MockTrial:
            def __init__(self, params):
                self.params = params
            def suggest_categorical(self, name, choices):
                return self.params.get(name, choices[0])
            def suggest_float(self, name, low, high, log=False):
                return self.params.get(name, (low + high) / 2)
            def suggest_int(self, name, low, high):
                return self.params.get(name, (low + high) // 2)
        
        mock_trial = MockTrial(best_params)
        config = TunableConfig(trial=mock_trial)
    else:
        config = TunableConfig()  # Use default optimized parameters
    
    # Train final model
    print("\nTraining final model...")
    results = train_fold1_only(train_df, config, test_df=test_df)
    
    if len(results) == 6:
        model, threshold, temperature, metrics, transformer, submission = results
    else:
        model, threshold, temperature, metrics, transformer = results
        submission = None
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    # Display results
    targets_met = 0
    total_targets = 6
    
    target_thresholds = {
        'f1_weighted': 0.70,
        'wcll': 0.64,
        'auc_roc': 0.80,
        'sensitivity': 0.90,
        'specificity': 0.90,
        'pr_auc': 0.90
    }
    
    for metric, value in metrics.items():
        if isinstance(value, dict) or metric in ['confusion_matrix', 'targets_met', 'target_achievement']:
            continue
            
        target_met = False
        if metric in target_thresholds:
            if metric == 'wcll':
                target_met = value <= target_thresholds[metric]
            else:
                target_met = value >= target_thresholds[metric]
                
            if target_met:
                targets_met += 1
                
            target_status = "✓" if target_met else "✗"
            target_value = f"(target {'<' if metric == 'wcll' else '>'} {target_thresholds[metric]}: {target_status})"
        else:
            target_value = ""
            
        print(f"  {metric}: {value:.4f} {target_value}")
    
    print(f"\nOverall target achievement: {targets_met}/{total_targets} metrics = {targets_met/total_targets:.0%}")
    
    return submission, metrics, model

# Function to train only fold 1 with improved configuration
def train_fold1_only(df, config, test_df=None):
    print("\n===== Training Optimized Model - Fold 1 =====\n")
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # Create folds
    if 'patient_id' in df.columns:
        skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
        split_iter = skf.split(df, df['label'], groups=df['patient_id'])
    else:
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        split_iter = skf.split(df, df['label'])
    
    # Get fold 1
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        if fold > 0:
            break
            
        print(f"\n{'='*50}\nTraining Optimized Model - Fold 1\n{'='*50}")
        
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train set: {len(train_fold)} samples, Val set: {len(val_fold)} samples")
        print(f"Train class distribution: {dict(train_fold['label'].value_counts())}")
        print(f"Val class distribution: {dict(val_fold['label'].value_counts())}")
        
        # Train model
        model, threshold, temperature, metrics, transformer, history = train_model(
            train_fold, val_fold, config, fold=0
        )
        
        model.history = history
        
        # Save model
        model_path = f"{config.SAVE_DIR}/optimized_fold1_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'threshold': threshold,
            'temperature': temperature,
            'metrics': metrics,
            'transformer': transformer,
            'history': history,
            'config': config
        }, model_path)
        
        print(f"\nOptimized Fold 1 Training Complete")
        print(f"Threshold: {threshold:.4f}, Temperature: {temperature:.4f}")
        
        # Make predictions if test data provided
        if test_df is not None:
            submission, predictions = predict_with_single_model(
                model, threshold, temperature, transformer, test_df, config
            )
            return model, threshold, temperature, metrics, transformer, submission
            
        return model, threshold, temperature, metrics, transformer

# Prediction function (same as before but with config parameter)
def predict_with_single_model(model, threshold, temperature, transformer, test_df, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_dataset = OptimizedDataset(
        test_df, 
        config.FEATURES_DIR,
        training=False,
        transformer=transformer
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False, 
        num_workers=0
    )
    
    all_probs = []
    all_image_ids = []
    
    with torch.no_grad():
        for features, _, image_ids, _ in tqdm(test_loader, desc="Making predictions"):
            features = features.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
            
            outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_image_ids.extend(list(image_ids))
    
    all_probs = np.array(all_probs)
    laa_probs = all_probs[:, 1]
    ce_probs = all_probs[:, 0]
    
    print(f"Using prediction threshold: {threshold:.4f}")
    predictions = (laa_probs > threshold).astype(int)
    
    # Prediction statistics
    ce_count = np.sum(predictions == 0)
    laa_count = np.sum(predictions == 1)
    total_count = len(predictions)
    
    print(f"Prediction balance: CE: {ce_count/total_count*100:.1f}%, LAA: {laa_count/total_count*100:.1f}%")
    
    # Create submission
    submission = pd.DataFrame({
        'image_id': all_image_ids,
        'label': np.array(['CE', 'LAA'])[predictions],
        'ce_prob': ce_probs,
        'laa_prob': laa_probs,
    })
    
    submission_path = f"{config.SAVE_DIR}/optimized_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    return submission, predictions

if __name__ == "__main__":
    # Run with hyperparameter tuning
    # main(tune_hyperparams=True, n_trials=50)
    
    # Or run with optimized defaults
    main(tune_hyperparams=False)