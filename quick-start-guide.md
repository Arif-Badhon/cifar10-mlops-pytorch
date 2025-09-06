# Quick Start Implementation Guide
## Get Your MLOps Pipeline Running in 30 Minutes

### Prerequisites Checklist

Before starting, ensure you have:
- [ ] Python 3.9+ installed
- [ ] Git installed and configured
- [ ] Docker installed (for containerization)
- [ ] Conda or virtualenv for environment management
- [ ] GitHub account (for CI/CD)
- [ ] Weights & Biases account (free at wandb.ai)

### Step-by-Step Setup

#### 1. Initialize Your Project (5 minutes)

```bash
# Create and navigate to project directory
mkdir cifar10-mlops-pipeline
cd cifar10-mlops-pipeline

# Initialize Git and DVC
git init
pip install dvc
dvc init

# Create the project structure
mkdir -p {config/{model,training,experiment},data/{raw,processed,external}}
mkdir -p {src/{data,training,evaluation,utils},models,tests/{unit,integration,smoke}}
mkdir -p {scripts,notebooks,deployment/{api,monitoring,kubernetes}}
mkdir -p {docs,artifacts/{models,plots,reports},.github/workflows}

# Create essential files
touch {README.md,requirements.txt,environment.yml,.gitignore}
touch config/config.yaml
```

#### 2. Set Up Environment (5 minutes)

Create `environment.yml`:
```yaml
name: cifar10-mlops
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch=2.0.1
  - torchvision=0.15.2
  - numpy=1.24.3
  - pandas=2.0.2
  - matplotlib=3.7.1
  - jupyter
  - pip
  - pip:
    - wandb==0.15.4
    - hydra-core==1.3.2
    - pytest==7.4.0
    - black==23.3.0
    - flake8==6.0.0
```

```bash
# Create and activate environment
conda env create -f environment.yml
conda activate cifar10-mlops

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

#### 3. Create Minimal Working Implementation (15 minutes)

**Create `config/config.yaml`:**
```yaml
project_name: cifar10-mlops
seed: 42
device: auto

model:
  name: resnet18
  num_classes: 10
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  validation_split: 0.2

paths:
  data: data/
  models: artifacts/models/
  logs: logs/

experiment:
  tracker: wandb
  project: cifar10-quick-start
  tags: [quick-start]
```

**Create `models/simple_cnn.py`:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
```

**Create `scripts/quick_train.py`:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.simple_cnn import SimpleCNN

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split training data
    val_split = config['training']['validation_split']
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model():
    # Load configuration
    config = load_config()
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Weights & Biases
    wandb.init(
        project=config['experiment']['project'],
        config=config,
        tags=config['experiment']['tags']
    )
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Create model
    model = SimpleCNN(num_classes=config['model']['num_classes'])
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * correct / total
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_accuracy': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_acc
        })
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(config['paths']['models']).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), f"{config['paths']['models']}/best_model.pth")
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    train_model()
```

#### 4. Add Basic Testing (3 minutes)

**Create `tests/test_basic.py`:**
```python
import pytest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.simple_cnn import SimpleCNN

def test_model_creation():
    model = SimpleCNN(num_classes=10)
    assert model is not None

def test_model_forward_pass():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 CIFAR-10 images
    
    output = model(x)
    
    assert output.shape == (4, 10)  # Batch size 4, 10 classes
    assert not torch.isnan(output).any()

def test_model_backward_pass():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Check that gradients were computed
    for param in model.parameters():
        assert param.grad is not None
```

#### 5. Run Your First Experiment (2 minutes)

```bash
# Install wandb and login
pip install wandb
wandb login  # Follow the prompts to get your API key from wandb.ai

# Run training
python scripts/quick_train.py

# Run tests
pytest tests/test_basic.py -v
```

### What You've Achieved

In just 30 minutes, you now have:

✅ **Working MLOps Pipeline**: Complete CIFAR-10 classification with experiment tracking  
✅ **Reproducible Training**: Seed management and consistent results  
✅ **Experiment Tracking**: Real-time metrics visualization with W&B  
✅ **Automated Testing**: Basic test suite with pytest  
✅ **Professional Structure**: Organized codebase following industry standards  

### Next Steps to Full Implementation

Now you can gradually implement the advanced features:

1. **Add DVC for Data Versioning** (15 min)
2. **Implement Advanced Model Architecture** (20 min)
3. **Set up CI/CD Pipeline** (30 min)
4. **Add Comprehensive Testing** (45 min)
5. **Implement Performance Profiling** (30 min)
6. **Add Docker Containerization** (20 min)

### Monitoring Your Progress

Your W&B dashboard will show:
- Training and validation loss curves
- Accuracy metrics over time
- System metrics and resource usage
- Model architecture visualization

Visit your project at: `https://wandb.ai/[your-username]/cifar10-quick-start`

### Troubleshooting Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size in config.yaml
training:
  batch_size: 16  # or 8
```

**Slow Training:**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"
```

**W&B Login Issues:**
```bash
# Re-login to wandb
wandb logout
wandb login
```

### Ready to Go Deeper?

This quick start gives you a solid foundation. The complete implementation in the main project document will teach you to build production-ready ML systems with advanced MLOps practices used by top tech companies.

**Happy Learning! 🚀**