import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from .model import SimpleCNN
from .dataset import get_classes

def get_model(path='/app/data/model.pth'):
    """Loads the model and ensures it is in evaluation mode."""
    device = torch.device("cpu")
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded weights from {path}")
    except FileNotFoundError:
        print("Warning: Model file not found. Initializing random weights.")
    
    model.eval() # CRITICAL: Turns off Dropout for consistent predictions
    return model

def transform_image(image_bytes):
    """
    Prepares the image to match CIFAR-10 format EXACTLY.
    1. Force Convert to RGB (fixes PNG alpha channel issues).
    2. Resize to 32x32 (squashes high-res images down).
    3. Normalize (scales pixel values to -1 to 1).
    """
    my_transforms = transforms.Compose([
        transforms.Resize((32, 32)),  # <--- FORCE RESIZE (Don't crop!)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0) # Add batch dimension (1, 3, 32, 32)

def predict(model, image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    
    # Get probabilities to see how confident it is
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, y_hat = outputs.max(1)
    
    class_index = y_hat.item()
    class_name = get_classes()[class_index]
    
    # Print debug info to the terminal logs
    print(f"Predicted: {class_name} (Index: {class_index})")
    print(f"Probabilities: {probabilities.detach().numpy()}")
    
    return class_name
