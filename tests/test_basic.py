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