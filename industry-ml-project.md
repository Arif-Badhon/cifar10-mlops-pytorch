# Industry-Ready Computer Vision MLOps Project
## Building a Production-Grade Image Classification Pipeline with Modern MLOps

### Project Overview

This project implements a **comprehensive, industry-ready computer vision pipeline** for CIFAR-10 image classification using state-of-the-art MLOps practices. Unlike typical academic exercises, this project demonstrates the full lifecycle of a production ML system, incorporating all aspects of Module 1: modern project structuring, reproducibility, experiment tracking, automated testing, and CI/CD.

**What You'll Build:**
- Production-grade PyTorch project with modular architecture
- Comprehensive experiment tracking and model versioning
- Automated data versioning and pipeline reproducibility
- Complete testing suite (unit tests, integration tests, smoke tests)
- CI/CD pipeline with automated linting, testing, and deployment
- Containerized deployment with monitoring capabilities

**Learning Objectives:**
1. Master modern PyTorch project organization and best practices
2. Implement production-ready configuration management with YAML
3. Establish complete reproducibility through environment and seed management
4. Build automated data versioning pipelines with DVC
5. Compare and implement experiment tracking (W&B vs MLflow)
6. Develop comprehensive ML testing strategies
7. Set up performance profiling and bottleneck analysis
8. Deploy CI/CD pipelines for ML projects

---

## Complete Project Structure

```
cifar10-mlops-pipeline/
├── README.md
├── LICENSE
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                          # DVC pipeline definition
├── params.yaml                       # DVC parameters
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── model-training.yml
├── config/
│   ├── __init__.py
│   ├── config.yaml                   # Main configuration
│   ├── model/
│   │   ├── resnet.yaml
│   │   └── vgg.yaml
│   ├── training/
│   │   ├── default.yaml
│   │   └── distributed.yaml
│   └── experiment/
│       ├── baseline.yaml
│       └── hyperopt.yaml
├── data/
│   ├── raw/                          # Raw CIFAR-10 data (DVC tracked)
│   ├── processed/                    # Preprocessed data (DVC tracked)
│   └── external/                     # External datasets
├── models/
│   ├── __init__.py
│   ├── base_model.py
│   ├── resnet.py
│   ├── vgg.py
│   └── custom_models.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── transforms.py
│   │   └── data_loader.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── metrics.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── reproducibility.py
│       ├── logging.py
│       ├── profiler.py
│       └── config_loader.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── profile_model.py
│   └── setup_project.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_data.py
│   │   └── test_training.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_end_to_end.py
│   └── smoke/
│       └── test_smoke.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   └── 03_results_analysis.ipynb
├── deployment/
│   ├── api/
│   │   ├── main.py
│   │   ├── models.py
│   │   └── routes.py
│   ├── monitoring/
│   │   └── monitor.py
│   └── kubernetes/
│       ├── deployment.yaml
│       └── service.yaml
├── docs/
│   ├── setup.md
│   ├── usage.md
│   └── api.md
└── artifacts/                        # Model artifacts (DVC tracked)
    ├── models/
    ├── plots/
    └── reports/
```

---

## Step-by-Step Implementation

### Phase 1: Project Setup and Environment Configuration

#### Step 1.1: Initialize the Project Structure

Create the complete directory structure and initialize version control:

```bash
# Create project directory
mkdir cifar10-mlops-pipeline
cd cifar10-mlops-pipeline

# Initialize Git and DVC
git init
dvc init

# Create directory structure
mkdir -p config/{model,training,experiment}
mkdir -p data/{raw,processed,external}
mkdir -p src/{data,training,evaluation,utils}
mkdir -p models tests/{unit,integration,smoke}
mkdir -p scripts notebooks deployment/{api,monitoring,kubernetes}
mkdir -p docs artifacts/{models,plots,reports}
mkdir -p .github/workflows
```

#### Step 1.2: Environment and Dependency Management

**Create `environment.yml`:**
```yaml
name: cifar10-mlops
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - torchvision=0.15.2
  - torchaudio=2.0.2
  - pytorch-cuda=11.8
  - numpy=1.24.3
  - pandas=2.0.2
  - scikit-learn=1.2.2
  - matplotlib=3.7.1
  - seaborn=0.12.2
  - jupyter
  - dvc[s3]
  - hydra-core
  - omegaconf
  - pip
  - pip:
    - wandb==0.15.4
    - mlflow==2.4.1
    - pytest==7.4.0
    - pytest-cov==4.1.0
    - black==23.3.0
    - flake8==6.0.0
    - isort==5.12.0
    - pre-commit==3.3.3
    - torchtest==0.5
    - pytorch-lightning==2.0.4
    - fastapi==0.100.0
    - uvicorn==0.22.0
```

**Create `pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cifar10-mlops"
version = "0.1.0"
description = "Production-ready CIFAR-10 classification with MLOps"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.10"

[tool.black]
line-length = 88
target-version = ['py310']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=src --cov-report=html --cov-report=term-missing"
```

#### Step 1.3: Configuration Management with Hydra

**Create `config/config.yaml`:**
```yaml
defaults:
  - model: resnet
  - training: default
  - experiment: baseline
  - _self_

# Global settings
project_name: cifar10-mlops
seed: 42
device: auto  # auto, cpu, cuda
num_workers: 4
pin_memory: true

# Paths
paths:
  data: data/
  models: artifacts/models/
  logs: logs/
  plots: artifacts/plots/

# Reproducibility
reproducibility:
  deterministic: true
  benchmark: false
  seed_workers: true

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Experiment tracking
experiment:
  tracker: wandb  # wandb, mlflow, both
  project: cifar10-classification
  tags: [baseline, resnet18]
  notes: "Baseline experiment with ResNet18"

# Monitoring
monitoring:
  profile_training: false
  log_gradients: false
  watch_model: false
```

**Create `config/model/resnet.yaml`:**
```yaml
# @package model
_target_: models.resnet.ResNet18
num_classes: 10
pretrained: false
dropout: 0.1

# Model-specific configurations
architecture:
  block_type: BasicBlock
  layers: [2, 2, 2, 2]
  channels: [64, 128, 256, 512]
  
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001
  weight_decay: 1e-4
  betas: [0.9, 0.999]

scheduler:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 30
  gamma: 0.1
```

**Create `config/training/default.yaml`:**
```yaml
# @package training
epochs: 100
batch_size: 128
validation_split: 0.2
early_stopping:
  patience: 10
  min_delta: 0.001
  
# Data augmentation
augmentation:
  horizontal_flip: 0.5
  rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  normalize:
    mean: [0.4914, 0.4822, 0.4465]
    std: [0.2023, 0.1994, 0.2010]

# Loss function
criterion:
  _target_: torch.nn.CrossEntropyLoss
  label_smoothing: 0.1

# Metrics to track
metrics:
  - accuracy
  - top5_accuracy
  - f1_score
  - precision
  - recall
```

### Phase 2: Data Pipeline and Versioning

#### Step 2.1: DVC Pipeline Configuration

**Create `dvc.yaml`:**
```yaml
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py
    deps:
      - scripts/prepare_data.py
      - config/config.yaml
    outs:
      - data/processed/train_dataset.pt
      - data/processed/val_dataset.pt
      - data/processed/test_dataset.pt
    params:
      - training.batch_size
      - training.validation_split
      
  train_model:
    cmd: python scripts/train.py experiment=baseline
    deps:
      - scripts/train.py
      - src/
      - models/
      - data/processed/
      - config/
    outs:
      - artifacts/models/best_model.pth
    metrics:
      - artifacts/metrics.json
    plots:
      - artifacts/plots/training_curves.json:
          cache: false
      - artifacts/plots/confusion_matrix.json:
          cache: false
    params:
      - model
      - training
      - seed
      
  evaluate_model:
    cmd: python scripts/evaluate.py
    deps:
      - scripts/evaluate.py
      - artifacts/models/best_model.pth
      - data/processed/test_dataset.pt
    metrics:
      - artifacts/evaluation_metrics.json
    plots:
      - artifacts/plots/evaluation_plots.json:
          cache: false
```

**Create `params.yaml`:**
```yaml
model:
  name: resnet18
  num_classes: 10
  dropout: 0.1

training:
  epochs: 100
  batch_size: 128
  learning_rate: 0.001
  validation_split: 0.2

data:
  dataset: cifar10
  num_workers: 4
  pin_memory: true

seed: 42
```

#### Step 2.2: Data Loading and Processing

**Create `src/data/datasets.py`:**
```python
"""Dataset implementations for CIFAR-10."""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CIFAR10Dataset:
    """CIFAR-10 dataset handler with MLOps best practices."""
    
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    def __init__(self, config):
        self.config = config
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self):
        """Training data augmentation pipeline."""
        aug_config = self.config.training.augmentation
        
        transform_list = [
            transforms.RandomHorizontalFlip(p=aug_config.horizontal_flip),
            transforms.RandomRotation(degrees=aug_config.rotation),
            transforms.ColorJitter(
                brightness=aug_config.color_jitter.brightness,
                contrast=aug_config.color_jitter.contrast,
                saturation=aug_config.color_jitter.saturation,
                hue=aug_config.color_jitter.hue
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config.normalize.mean,
                std=aug_config.normalize.std
            )
        ]
        
        return transforms.Compose(transform_list)
    
    def _get_val_transforms(self):
        """Validation/test data preprocessing pipeline."""
        aug_config = self.config.training.augmentation
        
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug_config.normalize.mean,
                std=aug_config.normalize.std
            )
        ])
    
    def get_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and split CIFAR-10 into train/val/test sets."""
        logger.info("Loading CIFAR-10 dataset...")
        
        # Download datasets
        full_train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.paths.data,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.config.paths.data,
            train=False,
            download=True,
            transform=self.val_transform
        )
        
        # Split training data into train/val
        val_split = self.config.training.validation_split
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # Use generator for reproducible splits
        generator = torch.Generator().manual_seed(self.config.seed)
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        # Update validation dataset transform
        val_dataset.dataset = torchvision.datasets.CIFAR10(
            root=self.config.paths.data,
            train=True,
            download=False,
            transform=self.val_transform
        )
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, "
                   f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders with proper configuration."""
        train_ds, val_ds, test_ds = self.get_datasets()
        
        # Worker initialization for reproducibility
        def worker_init_fn(worker_id):
            import numpy as np
            np.random.seed(self.config.seed + worker_id)
        
        # Create generator for reproducible data loading
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=worker_init_fn,
            generator=generator
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=worker_init_fn
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=worker_init_fn
        )
        
        return train_loader, val_loader, test_loader
```

### Phase 3: Model Architecture and Training

#### Step 3.1: Model Implementations

**Create `models/base_model.py`:**
```python
"""Base model class with MLOps best practices."""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseModel(nn.Module, ABC):
    """Base model class with common functionality."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self._setup_model()
        
    @abstractmethod
    def _setup_model(self):
        """Setup model architecture. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Must be implemented by subclasses."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
            'architecture': self.__class__.__name__
        }
    
    def initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
```

**Create `models/resnet.py`:**
```python
"""ResNet implementation for CIFAR-10."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class BasicBlock(nn.Module):
    """Basic ResNet block."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class ResNet18(BaseModel):
    """ResNet-18 for CIFAR-10 classification."""
    
    def __init__(self, num_classes=10, dropout=0.1):
        self.dropout = dropout
        super().__init__(num_classes)
        
    def _setup_model(self):
        """Setup ResNet-18 architecture."""
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)
        
        # Initialize weights
        self.initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Create a ResNet layer."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride, self.dropout))
            self.in_channels = out_channels * BasicBlock.expansion
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
```

#### Step 3.2: Training Infrastructure

**Create `src/training/trainer.py`:**
```python
"""Model trainer with experiment tracking and monitoring."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import logging
import time
import json
from pathlib import Path
import wandb
import mlflow
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ..utils.reproducibility import set_deterministic_behavior
from ..evaluation.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Production-ready model trainer with comprehensive monitoring."""
    
    def __init__(
        self, 
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: str = 'auto'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = self._setup_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.metrics_calculator = MetricsCalculator(num_classes=config.model.num_classes)
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            
        return device
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss function."""
        criterion_config = self.config.training.criterion
        criterion_class = getattr(nn, criterion_config._target_.split('.')[-1])
        criterion_params = {k: v for k, v in criterion_config.items() 
                           if k != '_target_'}
        return criterion_class(**criterion_params)
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        optimizer_config = self.config.model.optimizer
        optimizer_class = getattr(torch.optim, optimizer_config._target_.split('.')[-1])
        optimizer_params = {k: v for k, v in optimizer_config.items() 
                           if k != '_target_'}
        return optimizer_class(self.model.parameters(), **optimizer_params)
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        if hasattr(self.config.model, 'scheduler'):
            scheduler_config = self.config.model.scheduler
            scheduler_class = getattr(torch.optim.lr_scheduler, 
                                    scheduler_config._target_.split('.')[-1])
            scheduler_params = {k: v for k, v in scheduler_config.items() 
                               if k != '_target_'}
            return scheduler_class(self.optimizer, **scheduler_params)
        return None
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking (W&B and/or MLflow)."""
        tracker = self.config.experiment.tracker
        
        if tracker in ['wandb', 'both']:
            wandb.init(
                project=self.config.experiment.project,
                config=dict(self.config),
                tags=self.config.experiment.tags,
                notes=self.config.experiment.notes
            )
            
            if self.config.monitoring.watch_model:
                wandb.watch(self.model, log='all', log_freq=100)
        
        if tracker in ['mlflow', 'both']:
            mlflow.set_experiment(self.config.experiment.project)
            mlflow.start_run()
            mlflow.log_params(dict(self.config))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                logger.debug(f'Batch {batch_idx}/{len(self.train_loader)}, '
                           f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate_epoch(self) -> Dict[str, Any]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate additional metrics
        metrics = self.metrics_calculator.calculate_metrics(
            all_targets, all_predictions
        )
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            **metrics
        }
    
    def train(self) -> Dict[str, Any]:
        """Complete training loop."""
        logger.info("Starting training...")
        start_time = time.time()
        
        # Set deterministic behavior if configured
        if self.config.reproducibility.deterministic:
            set_deterministic_behavior(self.config.seed)
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train and validate
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, current_lr)
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self._save_checkpoint('best_model.pth', is_best=True)
            
            # Early stopping check
            if self._should_early_stop(val_metrics['accuracy']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save final training results
        results = {
            'best_val_accuracy': self.best_val_acc,
            'training_time': training_time,
            'epochs_trained': self.current_epoch + 1,
            'model_info': self.model.get_model_info()
        }
        
        self._save_training_results(results)
        return results
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, lr: float):
        """Log metrics for current epoch."""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_acc'].append(val_metrics['accuracy'])
        self.training_history['learning_rates'].append(lr)
        
        # Log to console
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%, "
            f"LR: {lr:.6f}"
        )
        
        # Log to experiment trackers
        metrics_dict = {
            'epoch': self.current_epoch,
            'train/loss': train_metrics['loss'],
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy'],
            'learning_rate': lr
        }
        
        # Add validation metrics
        for key, value in val_metrics.items():
            if key not in ['loss', 'accuracy']:
                metrics_dict[f'val/{key}'] = value
        
        if self.config.experiment.tracker in ['wandb', 'both']:
            wandb.log(metrics_dict)
            
        if self.config.experiment.tracker in ['mlflow', 'both']:
            for key, value in metrics_dict.items():
                mlflow.log_metric(key, value, step=self.current_epoch)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.paths.models) / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'training_history': self.training_history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logger.info(f"Saved best model with validation accuracy: {self.best_val_acc:.2f}%")
    
    def _should_early_stop(self, current_val_acc: float) -> bool:
        """Check if training should be stopped early."""
        early_stopping = self.config.training.early_stopping
        patience = early_stopping.patience
        min_delta = early_stopping.min_delta
        
        if len(self.training_history['val_acc']) < patience:
            return False
        
        # Check if there's improvement in the last 'patience' epochs
        recent_accs = self.training_history['val_acc'][-patience:]
        best_recent = max(recent_accs)
        
        return (current_val_acc - best_recent) < min_delta
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to file."""
        results_path = Path(self.config.paths.models) / 'training_results.json'
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Include training history
        results['training_history'] = self.training_history
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to {results_path}")
```

### Phase 4: Reproducibility and Testing

#### Step 4.1: Reproducibility Utilities

**Create `src/utils/reproducibility.py`:**
```python
"""Utilities for ensuring reproducibility in ML experiments."""
import torch
import numpy as np
import random
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible results across all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Random seed set to {seed}")

def set_deterministic_behavior(seed: int = 42, benchmark: bool = False) -> None:
    """Set completely deterministic behavior for reproducible results.
    
    Note: This may impact performance, especially on GPU.
    
    Args:
        seed: Random seed value
        benchmark: Whether to use CUDNN benchmark for performance optimization
    """
    set_seed(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark
    
    # Ensure deterministic algorithms
    torch.use_deterministic_algorithms(True)
    
    logger.info(f"Deterministic behavior enabled with seed {seed}")

def configure_worker_init_fn(seed: int) -> callable:
    """Create a worker initialization function for DataLoader reproducibility.
    
    Args:
        seed: Base seed for worker initialization
        
    Returns:
        Worker initialization function
    """
    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        
    return worker_init_fn

def get_reproducibility_info() -> dict:
    """Get information about current reproducibility settings.
    
    Returns:
        Dictionary with reproducibility information
    """
    return {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

def validate_reproducibility_environment() -> bool:
    """Validate that the environment is properly configured for reproducibility.
    
    Returns:
        True if environment is properly configured, False otherwise
    """
    issues = []
    
    # Check if PYTHONHASHSEED is set
    if not os.environ.get('PYTHONHASHSEED'):
        issues.append("PYTHONHASHSEED is not set")
    
    # Check CUDNN settings for GPU reproducibility
    if torch.cuda.is_available():
        if not torch.backends.cudnn.deterministic:
            issues.append("CUDNN deterministic mode is not enabled")
            
        if torch.backends.cudnn.benchmark:
            logger.warning("CUDNN benchmark is enabled - this may affect reproducibility")
    
    if issues:
        logger.error(f"Reproducibility issues found: {issues}")
        return False
    
    logger.info("Reproducibility environment validation passed")
    return True
```

#### Step 4.2: Comprehensive Testing Suite

**Create `tests/conftest.py`:**
```python
"""Pytest configuration and fixtures."""
import pytest
import torch
import numpy as np
from hydra import compose, initialize
from omegaconf import OmegaConf
import tempfile
from pathlib import Path

@pytest.fixture
def config():
    """Load test configuration."""
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="config")
        return cfg

@pytest.fixture
def small_config(config):
    """Create a small configuration for fast testing."""
    # Override config for testing
    config.training.epochs = 2
    config.training.batch_size = 8
    config.num_workers = 0  # No multiprocessing in tests
    return config

@pytest.fixture
def sample_data():
    """Generate sample CIFAR-10 like data."""
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels

@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(autouse=True)
def set_test_seed():
    """Set seed for all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
```

**Create `tests/unit/test_models.py`:**
```python
"""Unit tests for model implementations."""
import pytest
import torch
from models.resnet import ResNet18
from models.base_model import BaseModel

class TestResNet18:
    """Test ResNet18 model implementation."""
    
    def test_model_creation(self):
        """Test that model can be created with correct parameters."""
        model = ResNet18(num_classes=10, dropout=0.1)
        assert model.num_classes == 10
        assert isinstance(model, BaseModel)
    
    def test_forward_pass_shape(self, sample_data):
        """Test that forward pass produces correct output shape."""
        model = ResNet18(num_classes=10)
        images, _ = sample_data
        
        outputs = model(images)
        
        assert outputs.shape == (8, 10)  # batch_size, num_classes
        assert not torch.isnan(outputs).any()
        assert not torch.isinf(outputs).any()
    
    def test_backward_pass(self, sample_data):
        """Test that backward pass works correctly."""
        model = ResNet18(num_classes=10)
        images, labels = sample_data
        criterion = torch.nn.CrossEntropyLoss()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            assert param.grad is not None
    
    def test_model_parameters_update(self, sample_data):
        """Test that model parameters are updated during training."""
        model = ResNet18(num_classes=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        images, labels = sample_data
        
        # Store initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training step
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Check that parameters have changed
        for name, param in model.named_parameters():
            assert not torch.equal(initial_params[name], param), f"Parameter {name} did not update"
    
    def test_model_info(self):
        """Test model information method."""
        model = ResNet18(num_classes=10)
        info = model.get_model_info()
        
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
        assert 'model_size_mb' in info
        assert 'architecture' in info
        assert info['architecture'] == 'ResNet18'
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
    
    def test_deterministic_output(self, sample_data):
        """Test that model produces deterministic outputs with same seed."""
        images, _ = sample_data
        
        torch.manual_seed(42)
        model1 = ResNet18(num_classes=10)
        output1 = model1(images)
        
        torch.manual_seed(42)
        model2 = ResNet18(num_classes=10)
        output2 = model2(images)
        
        assert torch.allclose(output1, output2, atol=1e-6)

class TestModelInitialization:
    """Test model weight initialization."""
    
    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        model = ResNet18(num_classes=10)
        
        # Check that no weights are NaN or infinite
        for param in model.parameters():
            assert not torch.isnan(param).any()
            assert not torch.isinf(param).any()
    
    def test_different_initialization(self):
        """Test that different model instances have different initial weights."""
        model1 = ResNet18(num_classes=10)
        model2 = ResNet18(num_classes=10)
        
        # Models should have different initial weights
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            # Most parameters should be different (allowing for some identical values)
            different_elements = ~torch.isclose(param1, param2, atol=1e-6)
            assert different_elements.sum() / different_elements.numel() > 0.5
```

**Create `tests/smoke/test_smoke.py`:**
```python
"""Smoke tests for basic functionality."""
import pytest
import torch
from models.resnet import ResNet18
from src.data.datasets import CIFAR10Dataset
from src.training.trainer import ModelTrainer

class TestSmoke:
    """Smoke tests to verify basic functionality."""
    
    def test_model_instantiation_smoke(self):
        """Smoke test for model creation."""
        model = ResNet18(num_classes=10)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_data_loading_smoke(self, small_config):
        """Smoke test for data loading."""
        dataset = CIFAR10Dataset(small_config)
        train_loader, val_loader, test_loader = dataset.get_dataloaders()
        
        # Test that we can get a batch
        batch = next(iter(train_loader))
        images, labels = batch
        
        assert images.shape[0] == small_config.training.batch_size
        assert labels.shape[0] == small_config.training.batch_size
        assert images.shape[1:] == (3, 32, 32)
    
    def test_training_smoke(self, small_config, temp_dir):
        """Smoke test for training pipeline."""
        # Update paths to temp directory
        small_config.paths.models = str(temp_dir)
        small_config.paths.logs = str(temp_dir)
        
        # Create minimal training setup
        model = ResNet18(num_classes=10)
        dataset = CIFAR10Dataset(small_config)
        train_loader, val_loader, _ = dataset.get_dataloaders()
        
        # Modify config for very short training
        small_config.training.epochs = 1
        small_config.experiment.tracker = None  # Disable tracking for smoke test
        
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=small_config,
            device='cpu'
        )
        
        # This should not crash
        results = trainer.train()
        assert 'best_val_accuracy' in results
        assert results['epochs_trained'] == 1
    
    def test_prediction_smoke(self, sample_data):
        """Smoke test for model prediction."""
        model = ResNet18(num_classes=10)
        model.eval()
        
        images, _ = sample_data
        
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
        
        assert predictions.shape == (8,)  # batch_size
        assert all(0 <= p <= 9 for p in predictions)
```

### Phase 5: CI/CD Pipeline Setup

#### Step 5.1: GitHub Actions Configuration

**Create `.github/workflows/ci.yml`:**
```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10.0, 3.11]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install flake8 black isort pytest pytest-cov
    
    - name: Lint with flake8
      run: |
        # Stop build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # Exit-zero treats all errors as warnings
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Format check with black
      run: |
        black --check --diff .
    
    - name: Import sorting check with isort
      run: |
        isort --check-only --diff .
    
    - name: Type checking with mypy
      run: |
        pip install mypy
        mypy src/ --ignore-missing-imports || true

  test-suite:
    runs-on: ubuntu-latest
    needs: quality-checks
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run smoke tests
      run: |
        pytest tests/smoke/ -v --tb=short
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt

  build-test:
    runs-on: ubuntu-latest
    needs: [quality-checks, test-suite]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
    
    - name: Test installation
      run: |
        pip install dist/*.whl
        python -c "import src; print('Package imported successfully')"
```

**Create `.github/workflows/model-training.yml`:**
```yaml
name: Model Training Pipeline

on:
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Name of the experiment'
        required: true
        default: 'ci-experiment'
      config_name:
        description: 'Configuration name'
        required: true
        default: 'baseline'

jobs:
  train-model:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Fetch full history for DVC
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Set up DVC
      uses: iterative/setup-dvc@v1
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Configure DVC remote (if using cloud storage)
      run: |
        # dvc remote modify myremote access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
        # dvc remote modify myremote secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        echo "DVC remote configuration would go here"
    
    - name: Pull data
      run: |
        dvc pull || echo "No DVC data to pull"
    
    - name: Set up experiment tracking
      env:
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        echo "Experiment tracking configured"
    
    - name: Run training experiment
      run: |
        python scripts/train.py experiment=${{ github.event.inputs.config_name }} \
          experiment.name=${{ github.event.inputs.experiment_name }} \
          training.epochs=5  # Short training for CI
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts
        path: |
          artifacts/
          logs/
    
    - name: Push results to DVC
      run: |
        dvc add artifacts/
        git add artifacts.dvc
        git commit -m "Update model artifacts from CI training" || true
        # dvc push
        echo "DVC push would happen here"
```

#### Step 5.2: Pre-commit Hooks

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements
      - id: check-docstring-first

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black", "--filter-files"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--ignore-missing-imports]

  - repo: local
    hooks:
      - id: pytest-smoke
        name: pytest-smoke
        entry: pytest tests/smoke/
        language: system
        pass_filenames: false
        always_run: true
```

### Phase 6: Profiling and Monitoring

#### Step 6.1: Performance Profiling

**Create `src/utils/profiler.py`:**
```python
"""Performance profiling utilities for ML training."""
import torch
import time
import psutil
import GPUtil
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelProfiler:
    """Comprehensive model and training profiler."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("profiling_results")
        self.output_dir.mkdir(exist_ok=True)
        self.profile_data = {}
    
    def profile_model_inference(self, model: torch.nn.Module, input_tensor: torch.Tensor,
                               num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """Profile model inference performance."""
        model.eval()
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        # Synchronize before timing (important for GPU)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = torch.tensor(times)
        results = {
            'mean_inference_time': times.mean().item(),
            'std_inference_time': times.std().item(),
            'min_inference_time': times.min().item(),
            'max_inference_time': times.max().item(),
            'median_inference_time': times.median().item(),
            'throughput_fps': 1.0 / times.mean().item(),
            'batch_size': input_tensor.shape[0],
            'device': str(device)
        }
        
        logger.info(f"Inference profiling results: {results}")
        return results
    
    def profile_memory_usage(self, model: torch.nn.Module, 
                           input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Profile memory usage during forward and backward pass."""
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Initial memory state
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        
        # Forward pass
        model.train()
        output = model(input_tensor)
        
        if device.type == 'cuda':
            forward_memory = torch.cuda.memory_allocated()
            forward_peak = torch.cuda.max_memory_allocated()
        
        # Backward pass
        loss = output.sum()  # Dummy loss for profiling
        loss.backward()
        
        if device.type == 'cuda':
            backward_memory = torch.cuda.memory_allocated()
            backward_peak = torch.cuda.max_memory_allocated()
            
            results = {
                'initial_memory_mb': initial_memory / (1024**2),
                'forward_memory_mb': forward_memory / (1024**2),
                'backward_memory_mb': backward_memory / (1024**2),
                'forward_peak_mb': forward_peak / (1024**2),
                'backward_peak_mb': backward_peak / (1024**2),
                'memory_overhead_mb': (backward_memory - initial_memory) / (1024**2),
                'device': str(device)
            }
        else:
            # CPU memory profiling is more complex, simplified version
            results = {
                'device': 'cpu',
                'note': 'CPU memory profiling requires additional tools'
            }
        
        logger.info(f"Memory profiling results: {results}")
        return results
    
    @contextmanager
    def profile_training_step(self):
        """Context manager for profiling a training step."""
        # System metrics before
        cpu_before = psutil.cpu_percent()
        memory_before = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            gpu_before = GPUtil.getGPUs()[0].memoryUtil if GPUtil.getGPUs() else 0
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            
            # System metrics after
            cpu_after = psutil.cpu_percent()
            memory_after = psutil.virtual_memory().percent
            
            if torch.cuda.is_available():
                gpu_after = GPUtil.getGPUs()[0].memoryUtil if GPUtil.getGPUs() else 0
            
            step_time = end_time - start_time
            
            self.profile_data = {
                'step_time': step_time,
                'cpu_usage_before': cpu_before,
                'cpu_usage_after': cpu_after,
                'memory_usage_before': memory_before,
                'memory_usage_after': memory_after,
            }
            
            if torch.cuda.is_available():
                self.profile_data.update({
                    'gpu_memory_before': gpu_before,
                    'gpu_memory_after': gpu_after,
                })
    
    def profile_data_loading(self, dataloader, num_batches: int = 10) -> Dict[str, float]:
        """Profile data loading performance."""
        times = []
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            start_time = time.perf_counter()
            # Just access the data to force loading
            _ = batch[0].shape, batch[1].shape
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
        
        times = torch.tensor(times)
        results = {
            'mean_batch_load_time': times.mean().item(),
            'std_batch_load_time': times.std().item(),
            'total_data_load_time': times.sum().item(),
            'batches_profiled': num_batches
        }
        
        logger.info(f"Data loading profiling results: {results}")
        return results
    
    def run_comprehensive_profile(self, model: torch.nn.Module,
                                 dataloader, config: Any) -> Dict[str, Any]:
        """Run comprehensive profiling of model and training."""
        logger.info("Starting comprehensive profiling...")
        
        # Get sample batch for profiling
        sample_batch = next(iter(dataloader))
        sample_input = sample_batch[0]
        
        results = {}
        
        # Profile inference
        results['inference'] = self.profile_model_inference(model, sample_input)
        
        # Profile memory usage
        results['memory'] = self.profile_memory_usage(model, sample_input)
        
        # Profile data loading
        results['data_loading'] = self.profile_data_loading(dataloader)
        
        # Model information
        results['model_info'] = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        
        # System information
        results['system_info'] = {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_count': torch.cuda.device_count(),
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            results['system_info']['gpu_name'] = torch.cuda.get_device_name()
            results['system_info']['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Save results
        output_file = self.output_dir / 'comprehensive_profile.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comprehensive profiling results saved to {output_file}")
        return results

def profile_model_with_pytorch_profiler(model: torch.nn.Module, 
                                      dataloader, 
                                      num_steps: int = 10,
                                      output_dir: str = "profiler_output"):
    """Profile model using PyTorch's built-in profiler."""
    from torch.profiler import profile, record_function, ProfilerActivity
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= num_steps:
                break
                
            inputs = inputs.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            
            with record_function("forward"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            with record_function("backward"):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            prof.step()
    
    # Print key averages
    print("CPU time averages:")
    print(prof.key_averages().table(sort_by="cpu_time_total"))
    
    if torch.cuda.is_available():
        print("\nCUDA time averages:")
        print(prof.key_averages().table(sort_by="cuda_time_total"))
    
    logger.info(f"PyTorch profiler output saved to {output_path}")
```

#### Step 6.2: Training Script with Profiling

**Create `scripts/train.py`:**
```python
"""Main training script with comprehensive MLOps integration."""
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import torch
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.reproducibility import set_deterministic_behavior
from src.utils.profiler import ModelProfiler, profile_model_with_pytorch_profiler
from src.data.datasets import CIFAR10Dataset
from src.training.trainer import ModelTrainer
from models.resnet import ResNet18

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting CIFAR-10 MLOps Training Pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set reproducible behavior
    if cfg.reproducibility.deterministic:
        set_deterministic_behavior(
            seed=cfg.seed,
            benchmark=cfg.reproducibility.benchmark
        )
    
    # Setup device
    if cfg.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading CIFAR-10 dataset...")
    dataset = CIFAR10Dataset(cfg)
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    
    # Create model
    logger.info("Creating model...")
    model = ResNet18(
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout
    )
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model info: {model_info}")
    
    # Profile model if requested
    if cfg.monitoring.profile_training:
        logger.info("Running model profiling...")
        profiler = ModelProfiler(Path(cfg.paths.logs) / "profiling")
        profile_results = profiler.run_comprehensive_profile(model, train_loader, cfg)
        
        # Run PyTorch profiler for detailed analysis
        profile_model_with_pytorch_profiler(
            model, train_loader, num_steps=5,
            output_dir=str(Path(cfg.paths.logs) / "pytorch_profiler")
        )
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device
    )
    
    # Train model
    logger.info("Starting training...")
    results = trainer.train()
    
    # Log final results
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    logger.info(f"Training time: {results['training_time']:.2f} seconds")
    
    # Save final configuration
    config_save_path = Path(cfg.paths.models) / "final_config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    return results

if __name__ == "__main__":
    main()
```

---

## Conclusion

This **industry-ready MLOps project** provides you with a comprehensive foundation for building production-grade machine learning systems. By implementing this project, you will gain hands-on experience with:

### Key Learning Outcomes

1. **Modern Project Architecture**: Master the art of structuring ML projects for scalability and maintainability
2. **Configuration Management**: Implement flexible, hierarchical configuration systems with Hydra
3. **Reproducibility Engineering**: Ensure complete reproducibility across environments and experiments  
4. **Data Pipeline Engineering**: Build robust, versioned data pipelines with DVC
5. **Experiment Management**: Compare and implement both W&B and MLflow for comprehensive experiment tracking
6. **Quality Assurance**: Develop comprehensive testing strategies for ML systems
7. **Performance Optimization**: Profile and optimize ML training pipelines for production deployment
8. **DevOps Integration**: Implement complete CI/CD pipelines specifically designed for ML workflows

### Production Readiness Features

This project implements **industry-standard practices** that you'll encounter in professional ML environments:

- **Containerized Deployment** with Docker and Kubernetes
- **Automated Testing** including unit, integration, and smoke tests
- **Performance Monitoring** with comprehensive profiling and bottleneck analysis
- **Security Best Practices** with dependency scanning and secure CI/CD
- **Documentation Standards** with API documentation and usage guides
- **Scalability Considerations** with distributed training capabilities

### Next Steps for Advanced Implementation

Once you've mastered this foundation, consider extending the project with:

1. **Multi-GPU Training** with PyTorch Distributed Data Parallel
2. **Hyperparameter Optimization** with Optuna or Ray Tune  
3. **Model Serving** with FastAPI and automatic model deployment
4. **Monitoring & Alerting** with Prometheus and Grafana
5. **A/B Testing Infrastructure** for model comparison in production
6. **Data Drift Detection** for production model monitoring

This project serves as your **gateway to production ML engineering**, providing the solid foundation needed to build and deploy reliable, scalable machine learning systems in enterprise environments.