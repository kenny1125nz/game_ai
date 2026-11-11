import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from crnn import CRNN
from ocr_config import ALPHABET, MODEL_DIR
from online_dataset import OnlineGameOCRDataset, collate_fn, LabelTransformer, Generator
from pathlib import Path
import sys

parent_dir = str(Path(__file__).resolve().parent.parent)
# Add to sys.path
sys.path.append(parent_dir)
from data_utils import D2RTraininngContext

# --- Configuration ---
BATCH_SIZE = 64
LR = 0.0005
MAX_EPOCHS = 100
ITERATIONS_PER_EPOCH = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_model_and_optim(num_chars):
    """Initializes the CRNN, Optimizer, and Loss Criterion."""
    model = CRNN(num_chars=num_chars).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    return model, optimizer, criterion, scheduler


def load_checkpoint(model, optimizer, path):
    """Resumes training state from a saved file."""
    if not os.path.exists(path):
        return 0, float("inf")

    print(f"--- Loading Checkpoint: {path} ---")
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"] + 1, checkpoint.get("loss", float("inf"))


def train_one_epoch(model, loader, optimizer, criterion, epoch):
    """Runs one full pass over the defined epoch size."""
    model.train()
    total_loss = 0

    for i, (images, targets, target_lengths) in enumerate(loader):
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward
        logits = model(images)
        log_probs = logits.log_softmax(2)

        # CTC Lengths
        batch_size = images.size(0)
        input_lengths = torch.full((batch_size,), logits.size(0), dtype=torch.long).to(
            DEVICE
        )

        # Backward
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 100 == 0:
            print(
                f"Epoch {epoch} | Batch {i}/{ITERATIONS_PER_EPOCH} | Loss: {loss.item():.4f}"
            )

    return total_loss / len(loader)


def validate_and_preview(model, dataset, transformer):
    """Runs a single inference on a fresh sample to show progress."""
    model.eval()
    with torch.no_grad():
        test_img, _, _ = dataset[0]  # Grab a fresh random sample
        test_logits = model(test_img.unsqueeze(0).to(DEVICE))
        pred_indices = test_logits.argmax(2).squeeze().cpu().numpy()
        decoded = transformer.decode(pred_indices)
        print(f"Current Model Thought: '{decoded}'")
        return decoded


def main():
    # 1. Initialization
    transformer = LabelTransformer(ALPHABET)
    model, optimizer, criterion, scheduler = setup_model_and_optim(len(ALPHABET))

    # 2. Paths & Checkpoints
    checkpoint_path = os.path.join(MODEL_DIR, "d2r_ocr_model_checkpoint.pth")
    best_model_path = os.path.join(MODEL_DIR, "d2r_ocr_model.pth")
    start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)

    context = D2RTraininngContext("../data/clean_bg/")
    generator = Generator(context)
    # 3. Data Loader
    dataset = OnlineGameOCRDataset(
        epoch_size=BATCH_SIZE * ITERATIONS_PER_EPOCH,
        transformer=transformer,
        generator=generator,
        real_data_dir="./real_crops/",
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
    )

    # 4. Loop
    for epoch in range(start_epoch, MAX_EPOCHS):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, epoch)
        scheduler.step(avg_loss)

        # Persistence
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            checkpoint_path,
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"*** New Best: {best_loss:.4f} ***")

        validate_and_preview(model, dataset, transformer)

        if avg_loss < 0.03:
            print(f"model converged:{avg_loss}")
            break


if __name__ == "__main__":
    main()
