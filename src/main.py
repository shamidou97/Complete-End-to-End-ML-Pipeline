import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix

# Import our custom modules
from model import SimpleCNN
from dataset import get_dataloaders, get_classes

def train():
    # 1. Setup Device & Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    EPOCHS = 40  # Reduced from 100 (since we save the best, massive epochs aren't as critical)
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    DATA_PATH = '/app/data'
    CLASSES = get_classes()

    # 2. Get Data Loaders
    print("Preparing Data...")
    trainloader, valloader = get_dataloaders(DATA_PATH, BATCH_SIZE)

    # 3. Initialize Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ADDED: weight_decay=1e-4 (L2 Regularization) to punish large weights
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

    # Metrics storage
    train_losses = []
    val_losses = []
    min_val_loss = float('inf') # Track the best loss

    # 4. Training Loop
    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # --- Training Phase ---
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        avg_val_loss = val_running_loss / len(valloader)
        val_losses.append(avg_val_loss)

        # Log progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- CHECKPOINT: Save ONLY if validation improves ---
        if avg_val_loss < min_val_loss:
            print(f"   --> Validation Loss Improved ({min_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            min_val_loss = avg_val_loss
            save_path = os.path.join(DATA_PATH, 'model.pth')
            torch.save(model.state_dict(), save_path)

    print('Finished Training')
    print(f"Best Model was saved with Validation Loss: {min_val_loss:.4f}")

    # 6. Generate Plots
    generate_plots(train_losses, val_losses, valloader, model, device, CLASSES, DATA_PATH)

def generate_plots(train_losses, val_losses, valloader, model, device, classes, output_path):
    print("Generating analysis plots...")
    
    # Plot A: Loss vs Epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(val_losses, label='Validation Loss', color='orange', marker='x')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()

    # Plot B: Residuals / Confusion Matrix
    y_pred = []
    y_true = []
    
    # Reload the BEST model for the confusion matrix
    model.load_state_dict(torch.load(os.path.join(output_path, 'model.pth')))
    model.eval()
    
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Residuals/Errors)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_path, 'residuals_matrix.png'))
    plt.close()
    
    print("Plots saved.")

if __name__ == "__main__":
    train()
