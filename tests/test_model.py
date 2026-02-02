import torch
import pytest
from src.model import SimpleCNN

def test_model_structure():
    """
    Verify that the model accepts a 32x32 RGB image and outputs 10 classes.
    """
    model = SimpleCNN()
    
    # Create a dummy tensor representing a single image:
    # Batch Size=1, Channels=3 (RGB), Height=32, Width=32
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Pass it through the model
    output = model(dummy_input)
    
    # Assertions
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"
    print("Model structure test passed!")
