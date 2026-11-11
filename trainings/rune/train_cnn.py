import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from network.cnn_transformer_hybrid import RuneHybridNet, train_transforms
import sys

from data_loader import joined_load
from rune_config import config, TRAINING_DATA_DIR, MODEL_OUTPUT, ERROR_DATA_DIR
from utils import FocalLoss, diagnose_errors

# --- CONFIGURATION ---
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0001):
        self.patience = patience  # How many epochs to wait
        self.min_delta = min_delta  # Minimum change to count as an improvement
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc > (self.best_acc + self.min_delta):
            self.best_acc = val_acc
            self.counter = 0  # Reset if we improved
        else:
            self.counter += 1  # Increment if no significant improvement
            if self.counter >= self.patience:
                self.early_stop = True


def prepare_data_loaders():
    """Prepare and return train and validation data loaders."""
    image_size = config["image_size"][::-1]
    transform = train_transforms(image_size)

    full_dataset, sampler, classes, _ = joined_load(
        TRAINING_DATA_DIR, ERROR_DATA_DIR, transform
    )

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Check image range
    images, labels = next(iter(train_loader))
    print(f"Image Range: {images.min()} to {images.max()}")

    num_classes = len(classes)
    print(f"Loaded {len(full_dataset)} images across {num_classes} classes.")
    print(f"Classes: {classes}")

    return train_loader, val_loader, classes, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def validate_model(model, val_loader, device):
    """Validate the model and return accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def save_best_model(model, classes, accuracy, best_acc, model_file):
    """Save the model if it achieves the best accuracy."""
    if accuracy > best_acc:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
            },
            model_file,
        )
        print(f"Saved best model with accuracy: {accuracy:.2f}% to {model_file}")
        return accuracy
    return best_acc


def train_model(model, model_file):
    """Main training function."""
    print(f"Using device: {DEVICE}")

    # Prepare data
    train_loader, val_loader, classes, num_classes = prepare_data_loaders()

    # Setup optimizer, criterion, and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )
    criterion = FocalLoss(gamma=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    early_stopper = EarlyStopping(patience=3)
    # Training loop
    best_acc = 0.0
    for epoch in range(EPOCHS):
        # Train for one epoch
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        accuracy = validate_model(model, val_loader, DEVICE)

        early_stopper(accuracy)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print progress
        print(
            f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Val Acc: {accuracy:.2f}%"
        )

        # Save best model
        best_acc = save_best_model(model, classes, accuracy, best_acc, model_file)

    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")
    diagnose_errors(model, val_loader, classes, DEVICE)


# Add this call at the end of your training script:

if __name__ == "__main__":
    model_size = sys.argv[1]
    model_file = f"{MODEL_OUTPUT}{model_size}_hybrid_rune_classifier.pth"
    model = RuneHybridNet(num_classes=35, size=model_size).to(DEVICE)
    print(f"training {model_file}, model size: {model_size}")
    train_model(model, model_file)
