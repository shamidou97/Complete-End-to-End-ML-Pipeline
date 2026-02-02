# Expose key components for easier access
from .model import SimpleCNN
from .dataset import get_dataloaders, get_classes
from .inference import get_model, predict, transform_image
from .api import app
