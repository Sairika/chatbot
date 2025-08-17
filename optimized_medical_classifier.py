import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import h5py
import random
from torch.amp import autocast, GradScaler
import gc
import warnings
import json
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class OptimizedConfig:
    """Optimized configuration based on analysis"""
    # Data parameters
    FEATURE_DIM = 1024
    NUM_TILES = 16
    
    # Optimized training parameters
    BATCH_SIZE = 16          # Increased from 8
    LEARNING_RATE = 5e-5     # Increased from 1e-5
    WEIGHT_DECAY = 0.008     # Reduced from 0.015
    NUM_EPOCHS = 40          # Slightly increased
    PATIENCE = 12            # Reasonable patience
    
    # Model architecture - balanced complexity
    HIDDEN_DIM = 256
    OUTPUT_DIM = 2
    DROPOUT_RATE = 0.2       # Reduced from 0.3
    FEATURE_DROPOUT = 0.08   # Reduced from 0.15
    NUM_ATTENTION_HEADS = 4
    NUM_LAYERS = 2
    
    # Balanced class weights
    CE_CLASS_WEIGHT = 1.2
    LAA_CLASS_WEIGHT = 1.3
    
    # Loss settings
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.35
    USE_LABEL_SMOOTHING = True
    LABEL_SMOOTHING = 0.06   # Reduced from 0.1
    
    # Training enhancements
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0
    GRADIENT_ACCUMULATION = 4
    
    # Advanced techniques
    USE_EMA = True
    EMA_DECAY = 0.9995
    USE_TEMPERATURE_SCALING = True
    
    # Scheduler
    USE_COSINE_SCHEDULER = True
    WARMUP_EPOCHS = 3
    
    # Paths
    SAVE_DIR = "./optimized_models"
    FEATURES_DIR = "../input/train-attention-fusion-features-16-tiles"

def calculate_comprehensive_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive medical metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Medical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Advanced metrics
    auc_roc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = np.trapz(precision_curve, recall_curve)
    
    # Log loss with clipping
    eps = 1e-15
    y_probs_clipped = np.clip(y_probs, eps, 1 - eps)
    wcll = log_loss(y_true, y_probs_clipped)
    
    # G-mean and MCC
    g_mean = (sensitivity * specificity) ** 0.5 if sensitivity > 0 and specificity > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_weighted': f1_weighted,
        'auc_roc': auc_roc,
        'pr_auc': pr_auc,
        'wcll': wcll,
        'g_mean': g_mean,
        'mcc': mcc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }
    
    # Medical targets
    targets_met = {
        'sensitivity': sensitivity >= 0.90,
        'specificity': specificity >= 0.90,
        'f1_weighted': f1_weighted >= 0.70,
        'wcll': wcll <= 0.64,
        'pr_auc': pr_auc >= 0.90,
        'auc_roc': auc_roc >= 0.80
    }
    
    metrics['targets_met'] = targets_met
    metrics['target_achievement'] = sum(targets_met.values()) / len(targets_met)
    
    return metrics

class FeatureTransformer:
    """Efficient feature transformer"""
    def __init__(self, method='power'):
        self.method = method
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.fitted = False
        
    def fit(self, features):
        if features.shape[0] > 0:
            if len(features.shape) > 2:
                features = features.reshape(-1, features.shape[-1])
            self.transformer.fit(features)
            self.fitted = True
        
    def transform(self, features):
        if not self.fitted:
            return features
        original_shape = features.shape
        if len(original_shape) > 2:
            features = features.reshape(-1, original_shape[-1])
        transformed = self.transformer.transform(features)
        if len(original_shape) > 2:
            transformed = transformed.reshape(original_shape)
        return transformed

class MedicalAugmenter:
    """Medical-specific data augmentation"""
    def __init__(self, config):
        self.config = config
        self.noise_level = 0.02
        
    def __call__(self, features, training=True):
        if not training:
            return features
            
        features = features.clone()
        
        if random.random() < 0.5:
            # Light feature dropout
            if random.random() < 0.4:
                mask = torch.rand_like(features) > self.config.FEATURE_DROPOUT
                features = features * mask
            
            # Small gaussian noise
            if random.random() < 0.3:
                noise = torch.randn_like(features) * self.noise_level
                features = features + noise
                
        return features

class OptimizedDataset(Dataset):
    """Optimized dataset with efficient loading"""
    def __init__(self, df, data_dir, augmenter=None, training=True, transformer=None):
        self.df = df
        self.data_dir = data_dir
        self.augmenter = augmenter
        self.training = training
        self.transformer = transformer
        self.cache = {}
        
        # Patient mapping if available
        if 'patient_id' in self.df.columns:
            self.patient_ids = self.df['patient_id'].unique()
            self.patient_id_map = {pid: i for i, pid in enumerate(self.patient_ids)}
    
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row.image_id
        
        # Handle labels
        if 'label' in self.df.columns:
            label_value = row.label
            label = 0 if (isinstance(label_value, str) and label_value == "CE") else (0 if label_value == 0 else 1)
        else:
            label = -1
            
        # Patient ID
        patient_id = self.patient_id_map.get(row.get('patient_id', ''), -1) if 'patient_id' in self.df.columns else -1
    
        # Load features with caching
        if image_id in self.cache:
            features = self.cache[image_id]
        else:
            features = self._load_features(image_id)
            if len(self.cache) < 1000:  # Limit cache size
                self.cache[image_id] = features
        
        # Apply augmentation
        if self.augmenter and self.training:
            features = self.augmenter(features, self.training)
            
        return features, label, image_id, patient_id
    
    def _load_features(self, image_id):
        try:
            full_path = f"{self.data_dir}/{image_id}.h5"
            with h5py.File(full_path, 'r') as hdf5_file:
                tiles = []
                for i in range(OptimizedConfig.NUM_TILES):
                    if str(i) in hdf5_file:
                        feat = torch.tensor(hdf5_file[str(i)][:], dtype=torch.float32)
                        if len(feat.shape) > 1 and feat.shape[0] == 1:
                            feat = feat.squeeze(0)
                        tiles.append(feat)
                    else:
                        # Consistent placeholder
                        torch.manual_seed(hash(f"{image_id}_{i}") % 10000)
                        tiles.append(torch.zeros(OptimizedConfig.FEATURE_DIM) + 
                                   torch.randn(OptimizedConfig.FEATURE_DIM) * 0.01)
                
                features = torch.stack(tiles, dim=0)
                
                # Apply transformation
                if self.transformer and self.transformer.fitted:
                    features_np = features.numpy()
                    features_np = self.transformer.transform(features_np)
                    features = torch.tensor(features_np, dtype=torch.float32)
                    
                return features
                
        except Exception as e:
            print(f"Error loading {image_id}: {str(e)}")
            # Return consistent random features
            torch.manual_seed(hash(image_id) % 10000)
            return torch.randn((OptimizedConfig.NUM_TILES, OptimizedConfig.FEATURE_DIM)) * 0.01

def create_transformer(train_df, data_dir, sample_size=800):
    """Create and fit feature transformer efficiently"""
    print("Creating feature transformer...")
    
    transformer = FeatureTransformer()
    all_features = []
    
    # Balanced sampling
    ce_samples = train_df[train_df['label'] == 0].sample(n=min(sample_size//2, len(train_df[train_df['label'] == 0])), random_state=42)
    laa_samples = train_df[train_df['label'] == 1].sample(n=min(sample_size//2, len(train_df[train_df['label'] == 1])), random_state=42)
    sample_df = pd.concat([ce_samples, laa_samples])
    
    for idx in tqdm(range(len(sample_df)), desc="Fitting transformer"):
        image_id = sample_df.iloc[idx].image_id
        try:
            full_path = f"{data_dir}/{image_id}.h5"
            with h5py.File(full_path, 'r') as hdf5_file:
                for i in range(OptimizedConfig.NUM_TILES):
                    if str(i) in hdf5_file:
                        feat = np.array(hdf5_file[str(i)][:])
                        if len(feat.shape) > 1 and feat.shape[0] == 1:
                            feat = feat.squeeze(0)
                        all_features.append(feat)
        except:
            continue
    
    if all_features:
        features_array = np.vstack(all_features)
        transformer.fit(features_array)
        print(f"Transformer fitted on {len(features_array)} features")
    
    return transformer

class OptimizedModel(nn.Module):
    """Streamlined transformer model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(config.FEATURE_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE * 0.5)
        )
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, config.NUM_TILES, config.HIDDEN_DIM) * 0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.HIDDEN_DIM, config.NUM_ATTENTION_HEADS, config.DROPOUT_RATE)
            for _ in range(config.NUM_LAYERS)
        ])
        
        # Global attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 4),
            nn.LayerNorm(config.HIDDEN_DIM // 4),
            nn.GELU(),
            nn.Linear(config.HIDDEN_DIM // 4, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_RATE * 0.3),
            nn.Linear(config.HIDDEN_DIM // 2, config.OUTPUT_DIM)
        )
        
    def forward(self, x):
        batch_size, num_tiles, feat_dim = x.shape
        
        # Embedding
        x_flat = x.view(-1, feat_dim)
        embedded = self.embedding(x_flat)
        embedded = embedded.view(batch_size, num_tiles, -1)
        
        # Add position embeddings
        embedded = embedded + self.pos_embedding[:, :num_tiles, :]
        
        # Transformer blocks
        for block in self.transformer_blocks:
            embedded = block(embedded)
        
        # Global attention pooling
        attention_weights = F.softmax(self.attention_pool(embedded), dim=1)
        pooled = torch.sum(embedded * attention_weights, dim=1)
        
        # Classification
        return self.classifier(pooled)

class TransformerBlock(nn.Module):
    """Efficient transformer block"""
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate * 0.3,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate * 0.5)
        
    def forward(self, x):
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout1(attn_out)
        
        # MLP
        normed = self.norm2(x)
        mlp_out = self.mlp(normed)
        x = x + self.dropout2(mlp_out)
        
        return x

class FocalLoss(nn.Module):
    """Optimized focal loss with label smoothing"""
    def __init__(self, config):
        super().__init__()
        self.gamma = config.FOCAL_GAMMA
        self.alpha = config.FOCAL_ALPHA
        self.smoothing = config.LABEL_SMOOTHING if config.USE_LABEL_SMOOTHING else 0.0
        self.ce_weight = config.CE_CLASS_WEIGHT
        self.laa_weight = config.LAA_CLASS_WEIGHT
        
    def forward(self, logits, targets):
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
        
        # One-hot with label smoothing
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        if self.smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.smoothing) + self.smoothing / 2
        
        # Focal loss components
        p_t = torch.sum(probs * targets_one_hot, dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t)
        
        # Class weights
        class_weights = torch.where(targets == 0, self.ce_weight, self.laa_weight)
        
        # Alpha weighting
        alpha_weight = torch.where(targets == 0, 1 - self.alpha, self.alpha)
        
        loss = class_weights * alpha_weight * focal_weight * ce_loss
        return loss.mean()

class EMA:
    """Exponential Moving Average"""
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadows
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                
    def apply(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class TemperatureScaling:
    """Temperature scaling for calibration"""
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, logits, targets):
        from scipy.optimize import minimize
        
        def objective(temp):
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=1)
            nll = F.nll_loss(torch.log(probs + 1e-8), targets)
            return nll.item()
        
        result = minimize(objective, x0=1.0, bounds=[(0.1, 5.0)], method='L-BFGS-B')
        self.temperature = result.x[0]
        
    def __call__(self, logits):
        return logits / self.temperature

class BalancedSampler:
    """Balanced sampling for imbalanced data"""
    def __init__(self, labels, class_weights=None):
        self.labels = np.array(labels)
        self.class_counts = np.bincount(self.labels)
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
        
        # Calculate weights
        weights = np.zeros_like(self.labels, dtype=np.float32)
        for i, count in enumerate(self.class_counts):
            mask = self.labels == i
            weight = (1.0 / count) * self.class_weights[i] if count > 0 else 0
            weights[mask] = weight
        
        self.weights = weights / weights.sum()
        
    def __iter__(self):
        indices = np.random.choice(len(self.labels), size=len(self.labels), replace=True, p=self.weights)
        return iter(indices)
        
    def __len__(self):
        return len(self.labels)

def find_optimal_threshold(y_true, y_probs, n_points=101):
    """Find optimal threshold balancing medical requirements"""
    thresholds = np.linspace(0.1, 0.9, n_points)
    best_score = -1
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        metrics = calculate_comprehensive_metrics(y_true, y_pred, y_probs)
        
        # Multi-objective score prioritizing medical requirements
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        g_mean = metrics['g_mean']
        f1_weighted = metrics['f1_weighted']
        
        # Balance penalty
        balance_penalty = abs(sensitivity - specificity) * 0.1
        
        # Medical score with targets
        medical_score = (sensitivity * 0.35 + specificity * 0.35 + g_mean * 0.3)
        total_score = medical_score + f1_weighted * 0.2 - balance_penalty
        
        # Bonus for meeting constraints
        if sensitivity >= 0.85 and specificity >= 0.85:
            total_score *= 1.15
        
        if total_score > best_score:
            best_score = total_score
            best_threshold = threshold
            best_metrics = metrics
    
    return best_threshold, best_metrics

def train_model(train_df, val_df, config):
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Create transformer
    transformer = create_transformer(train_df, config.FEATURES_DIR)
    
    # Create datasets
    augmenter = MedicalAugmenter(config)
    
    train_dataset = OptimizedDataset(train_df, config.FEATURES_DIR, augmenter, True, transformer)
    val_dataset = OptimizedDataset(val_df, config.FEATURES_DIR, None, False, transformer)
    
    # Create balanced sampler
    labels = [0 if (isinstance(row.label, str) and row.label == "CE") else (0 if row.label == 0 else 1) 
              for _, row in train_df.iterrows()]
    
    class_weights = {0: config.CE_CLASS_WEIGHT, 1: config.LAA_CLASS_WEIGHT}
    sampler = BalancedSampler(labels, class_weights)
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0)
    
    # Model and training components
    model = OptimizedModel(config).to(device)
    criterion = FocalLoss(config) if config.USE_FOCAL_LOSS else nn.CrossEntropyLoss(
        weight=torch.tensor([config.CE_CLASS_WEIGHT, config.LAA_CLASS_WEIGHT], device=device)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Scheduler
    if config.USE_COSINE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=len(train_loader) * config.NUM_EPOCHS // (3 * config.GRADIENT_ACCUMULATION), eta_min=config.LEARNING_RATE / 50
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # EMA and temperature scaling
    ema = EMA(model, config.EMA_DECAY) if config.USE_EMA else None
    temp_scaler = TemperatureScaling() if config.USE_TEMPERATURE_SCALING else None
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training tracking
    best_target_achievement = 0
    best_metrics = {}
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_target_achievement': [], 'val_g_mean': []}
    
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch_idx, (features, labels, _, _) in enumerate(pbar):
            features, labels = features.to(device), labels.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
                loss = criterion(outputs, labels) / config.GRADIENT_ACCUMULATION
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION == 0:
                if config.USE_GRADIENT_CLIPPING:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_VALUE)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if config.USE_COSINE_SCHEDULER:
                    scheduler.step()
                
                if ema:
                    ema.update()
            
            train_loss += loss.item() * config.GRADIENT_ACCUMULATION
            pbar.set_postfix({'loss': f"{loss.item() * config.GRADIENT_ACCUMULATION:.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        if ema:
            ema.apply()
        
        model.eval()
        val_loss = 0
        val_probs = []
        val_labels = []
        val_logits = []
        
        with torch.no_grad():
            for features, labels, _, _ in val_loader:
                features, labels = features.to(device), labels.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_logits.append(outputs.cpu())
                
                probs = F.softmax(outputs, dim=1)
                val_probs.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        if ema:
            ema.restore()
        
        val_loss /= len(val_loader)
        
        # Temperature scaling
        if temp_scaler and epoch >= 2:
            all_logits = torch.cat(val_logits)
            temp_scaler.fit(all_logits, torch.tensor(val_labels))
        
        # Find optimal threshold and calculate metrics
        threshold, val_metrics = find_optimal_threshold(val_labels, val_probs)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_target_achievement'].append(val_metrics['target_achievement'])
        history['val_g_mean'].append(val_metrics['g_mean'])
        
        # Print progress
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Metrics - G-Mean: {val_metrics['g_mean']:.4f}, Target Achievement: {val_metrics['target_achievement']:.2%}")
        print(f"Sensitivity: {val_metrics['sensitivity']:.4f}, Specificity: {val_metrics['specificity']:.4f}")
        
        # Model saving
        target_improvement = val_metrics['target_achievement'] > best_target_achievement * 1.01
        g_mean_improvement = val_metrics['g_mean'] > best_metrics.get('g_mean', 0) * 1.01
        
        if target_improvement or g_mean_improvement or (epoch >= 5 and not best_metrics):
            best_target_achievement = max(best_target_achievement, val_metrics['target_achievement'])
            best_metrics = val_metrics.copy()
            patience_counter = 0
            
            # Save model
            os.makedirs(config.SAVE_DIR, exist_ok=True)
            save_dict = {
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
                'threshold': threshold,
                'temperature': temp_scaler.temperature if temp_scaler else 1.0,
                'transformer': transformer,
                'config': config,
                'history': history
            }
            
            if ema:
                save_dict['ema_shadows'] = ema.shadow
            
            torch.save(save_dict, f"{config.SAVE_DIR}/best_model.pth")
            print(f"✓ Model saved! Target achievement: {val_metrics['target_achievement']:.2%}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.PATIENCE and epoch >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Scheduler step for ReduceLROnPlateau
        if not config.USE_COSINE_SCHEDULER:
            scheduler.step(val_loss)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Load best model
    checkpoint = torch.load(f"{config.SAVE_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if ema and 'ema_shadows' in checkpoint:
        ema.shadow = checkpoint['ema_shadows']
        ema.apply()
    
    return model, checkpoint['threshold'], checkpoint['temperature'], best_metrics, transformer, history

def train_single_fold(df, config, fold_idx=0):
    """Train on a single fold"""
    print(f"\n{'='*60}")
    print(f"TRAINING OPTIMIZED MEDICAL CLASSIFIER - FOLD {fold_idx+1}")
    print(f"{'='*60}")
    
    # Create fold split
    if 'patient_id' in df.columns:
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(df, df['label'], groups=df['patient_id']))
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(df, df['label']))
    
    train_idx, val_idx = splits[fold_idx]
    train_fold = df.iloc[train_idx].reset_index(drop=True)
    val_fold = df.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train: {len(train_fold)} samples, Validation: {len(val_fold)} samples")
    print(f"Train distribution: {dict(train_fold['label'].value_counts())}")
    print(f"Val distribution: {dict(val_fold['label'].value_counts())}")
    
    # Train model
    model, threshold, temperature, metrics, transformer, history = train_model(train_fold, val_fold, config)
    
    return model, threshold, temperature, metrics, transformer, history

def predict_test(model, threshold, temperature, transformer, test_df, config):
    """Make predictions on test data"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_dataset = OptimizedDataset(test_df, config.FEATURES_DIR, None, False, transformer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=0)
    
    all_probs = []
    all_image_ids = []
    
    print("Making predictions...")
    with torch.no_grad():
        for features, _, image_ids, _ in tqdm(test_loader):
            features = features.to(device)
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(features)
            
            # Apply temperature scaling
            outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_image_ids.extend(image_ids)
    
    # Convert to arrays
    all_probs = np.array(all_probs)
    ce_probs = all_probs[:, 0]
    laa_probs = all_probs[:, 1]
    
    # Apply threshold
    predictions = (laa_probs > threshold).astype(int)
    
    # Statistics
    ce_count = np.sum(predictions == 0)
    laa_count = np.sum(predictions == 1)
    total = len(predictions)
    
    print(f"\nPrediction Summary:")
    print(f"Threshold: {threshold:.4f}")
    print(f"CE predictions: {ce_count} ({ce_count/total*100:.1f}%)")
    print(f"LAA predictions: {laa_count} ({laa_count/total*100:.1f}%)")
    
    # Create submission
    submission = pd.DataFrame({
        'image_id': all_image_ids,
        'label': ['CE' if p == 0 else 'LAA' for p in predictions],
        'ce_prob': ce_probs,
        'laa_prob': laa_probs
    })
    
    return submission

def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Target Achievement
    axes[0, 1].plot(history['val_target_achievement'])
    axes[0, 1].set_title('Target Achievement')
    axes[0, 1].set_ylabel('Percentage')
    
    # G-Mean
    axes[1, 0].plot(history['val_g_mean'])
    axes[1, 0].set_title('G-Mean (Sensitivity × Specificity)^0.5')
    
    # Combined view
    axes[1, 1].plot(history['val_target_achievement'], label='Target Achievement')
    axes[1, 1].plot(history['val_g_mean'], label='G-Mean')
    axes[1, 1].set_title('Key Metrics')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training plots saved to {save_path}")

def main():
    """Main function"""
    print("🏥 OPTIMIZED MEDICAL IMAGE CLASSIFIER")
    print("="*50)
    
    # Load data
    train_df = pd.read_csv("../input/mayo-clinic-strip-ai/train.csv")
    test_df = pd.read_csv("../input/mayo-clinic-strip-ai/test.csv")
    
    print(f"📊 Data: {len(train_df)} train samples, {len(test_df)} test samples")
    
    # Prepare data
    if 'patient_id' not in train_df.columns:
        train_df['patient_id'] = train_df['image_id'].apply(lambda x: x.split('_')[0] if '_' in x else x)
    
    if train_df['label'].dtype == 'object':
        label_map = {'CE': 0, 'LAA': 1}
        train_df['original_label'] = train_df['label']
        train_df['label'] = train_df['label'].map(label_map)
    
    # Configuration
    config = OptimizedConfig()
    
    # Train model
    model, threshold, temperature, metrics, transformer, history = train_single_fold(train_df, config, fold_idx=0)
    
    # Results summary
    print(f"\n{'='*60}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*60}")
    
    targets_met = 0
    total_targets = 6
    target_thresholds = {
        'sensitivity': 0.90, 'specificity': 0.90, 'f1_weighted': 0.70,
        'wcll': 0.64, 'pr_auc': 0.90, 'auc_roc': 0.80
    }
    
    for metric, value in metrics.items():
        if metric in target_thresholds:
            if metric == 'wcll':
                target_met = value <= target_thresholds[metric]
                comparison = '<='
            else:
                target_met = value >= target_thresholds[metric]
                comparison = '>='
            
            if target_met:
                targets_met += 1
            
            status = "✅" if target_met else "❌"
            print(f"{metric:12}: {value:.4f} (target {comparison} {target_thresholds[metric]}) {status}")
    
    print(f"\n🏆 Overall Target Achievement: {targets_met}/{total_targets} = {targets_met/total_targets:.0%}")
    print(f"📈 G-Mean (Balance): {metrics['g_mean']:.4f}")
    
    # Make predictions
    submission = predict_test(model, threshold, temperature, transformer, test_df, config)
    
    # Save results
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    submission_path = f"{config.SAVE_DIR}/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"💾 Submission saved to {submission_path}")
    
    # Plot training history
    plot_training_history(history, f"{config.SAVE_DIR}/training_history.png")
    
    # Save final summary
    summary = {
        'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in metrics.items() if not isinstance(v, dict)},
        'threshold': float(threshold),
        'temperature': float(temperature),
        'target_achievement': float(metrics['target_achievement']),
        'targets_met': f"{targets_met}/{total_targets}"
    }
    
    with open(f"{config.SAVE_DIR}/results_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✨ Training complete! Check {config.SAVE_DIR} for all outputs.")
    
    return submission, metrics, model

if __name__ == "__main__":
    main()