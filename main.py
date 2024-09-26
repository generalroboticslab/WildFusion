import os
import time
import argparse 
from src.models import MultiModalNetwork
from src.data_loader import CustomDataset, custom_collate_fn
from src.losses import EikonalLoss, ConfidenceLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
import config as cfg

# Set up argument parser for command-line flags
parser = argparse.ArgumentParser(description="Train the model with or without pretrained weights.")
parser.add_argument('--scratch', action='store_true', help='Train the model from scratch (without loading pretrained model).')
parser.add_argument('--skip_plot', action='store_true', help='Skip saving the loss plots.')  # New flag for skipping plot saving
args = parser.parse_args()

# Set the seed for random, NumPy, and PyTorch
random.seed(cfg.SEED)
np.random.seed(cfg.SEED)
torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

def plot_losses(epoch):
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(training_losses_sdf, label="Training Loss SDF")
    plt.plot(validation_losses_sdf, label="Validation Loss SDF")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for SDF")

    plt.subplot(2, 3, 2)
    plt.plot(training_losses_confidence, label="Training Loss Confidence")
    plt.plot(validation_losses_confidence, label="Validation Loss Confidence")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for Confidence")

    plt.subplot(2, 3, 3)
    plt.plot(training_losses_semantics, label="Training Loss Semantics")
    plt.plot(validation_losses_semantics, label="Validation Loss Semantics")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for Semantics")

    plt.subplot(2, 3, 4)
    plt.plot(training_losses_color, label="Training Loss Color")
    plt.plot(validation_losses_color, label="Validation Loss Color")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for Color")

    plt.subplot(2, 3, 5)
    plt.plot(training_losses_trav, label="Training Loss Traversability")
    plt.plot(validation_losses_trav, label="Validation Loss Traversability")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss for Traversability")

    plt.tight_layout()

    # Save the plot
    plt.savefig(cfg.INDIVIDUAL_LOSS_PLOT_PATH)
    plt.close()  # Close the figure to avoid memory warnings
    print(f"Individual loss plot saved to: {cfg.INDIVIDUAL_LOSS_PLOT_PATH}")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")

    # Save the plot
    plt.savefig(cfg.LOSS_PLOT_PATH)
    plt.close()  # Close the figure to avoid memory warnings
    print(f"Plot saved to: {cfg.LOSS_PLOT_PATH}")

# Ensure the save directory exists
if not os.path.exists(cfg.SAVE_DIR):
    os.makedirs(cfg.SAVE_DIR)

# Initialize loss trackers
training_losses = []
validation_losses = []

training_losses_semantics = []
training_losses_color = []
training_losses_confidence = []
training_losses_sdf = []
training_losses_trav = []

validation_losses_semantics = []
validation_losses_color = []
validation_losses_confidence = []
validation_losses_sdf = []
validation_losses_trav = []

def train_val(model, dataloader, val_dataloader, epochs, lr, checkpoint_path, best_model_path, use_pretrained=False, pretrained_model_path=None):
    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': model.module.sdf_fcn.parameters(), 'weight_decay': 1e-4},
        {'params': model.module.confidence_fcn.parameters(), 'weight_decay': 1e-4},
        {'params': model.module.semantic_fcn.parameters(), 'lr': 5e-6, 'weight_decay': 1e-5},
        {'params': model.module.color_fcn.parameters(), 'weight_decay': 1e-4},
        {'params': model.module.traversability_fc.parameters(), 'weight_decay': 1e-4}
    ], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.LR_DECAY_FACTOR, patience=cfg.PATIENCE)

    if os.path.exists(checkpoint_path) and use_pretrained:
        print(f"Checkpoint found at {checkpoint_path}. Resuming training.")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
    else:
        start_epoch = 0
        best_loss = float('inf')

    # Start timing
    start_time = time.time()

    criterion_mse = nn.MSELoss()
    criterion_ce_semantics = nn.CrossEntropyLoss(ignore_index=0)
    criterion_ce_color = nn.CrossEntropyLoss(ignore_index=312)
    criterion_huber = nn.SmoothL1Loss()
    eikonal_loss = EikonalLoss()
    criterion_l_conf = ConfidenceLoss(alpha=1.0, beta=5.0)

    best_sdf_val_loss = float('inf')
    epochs_no_improve_sdf = 0

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        # Skip plot if --skip_plot flag is passed
        if (epoch + 1) % cfg.PLOT_INTERVAL == 0 and not args.skip_plot:
            plot_losses(epoch + 1)

        for batch in dataloader:
            locations = batch['locations'].to(device)
            point_clouds = batch['point_clouds'].to(device)
            gt_sdf = batch['gt_sdf'].to(device)
            gt_confidence = batch['gt_confidence'].to(device)
            gt_semantics = batch['gt_semantics'].to(device)
            gt_color = batch['gt_color'].to(device)
            audio_data = batch['mel_spectrograms'].to(device)
            traversability = batch['gt_travs'].to(device)

            locations.requires_grad_(True)

            optimizer.zero_grad()

            if len(point_clouds.shape) == 4:
                batch_size, scans_per_batch, channels, num_points = point_clouds.shape
                point_clouds = point_clouds.view(batch_size * scans_per_batch, channels, num_points)

            batch_size, scans_per_batch, channels, height, width = audio_data.shape
            audio_data = audio_data.view(batch_size * scans_per_batch, channels, height, width)

            if len(locations.shape) == 2:
                num_locations = locations.shape[0] // batch_size
                locations = locations.view(batch_size, num_locations, 3)

            # Predictions from model
            preds_sdf, preds_confidence, preds_semantics, preds_color_logits, preds_trav = model(locations, point_clouds, audio_data)
            preds_sdf = preds_sdf.view(-1)
            preds_confidence = preds_confidence.view(-1)
            preds_trav = preds_trav.view(batch_size, -1)

            gt_sdf = gt_sdf.view(-1)
            gt_confidence = gt_confidence.view(-1)
            preds_trav_mean = preds_trav.mean(dim=1)

            # Loss calculations
            eikonal_loss_value = eikonal_loss(preds_sdf, locations)
            loss_eikonal = cfg.WEIGHT_EL * eikonal_loss_value
            loss_sdf = cfg.WEIGHT_SDF * criterion_huber(preds_sdf, gt_sdf)
            loss_confidence = cfg.WEIGHT_CONFIDENCE * criterion_l_conf(preds_confidence, gt_confidence)
            loss_semantics = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics.view(-1, cfg.CLASSES), gt_semantics.long().view(-1))
            preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
            gt_color = gt_color.view(-1)
            loss_color = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
            loss_trav = cfg.WEIGHT_TRAV * criterion_mse(preds_trav_mean, traversability.mean(dim=1))
            
            # Total loss
            total_loss = loss_sdf + loss_confidence + loss_semantics + loss_color + loss_trav + loss_eikonal
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        average_epoch_loss = epoch_loss / len(dataloader)
        
        training_losses.append(average_epoch_loss)
        training_losses_semantics.append(loss_semantics.item())
        training_losses_color.append(loss_color.item())
        training_losses_confidence.append(loss_confidence.item())
        training_losses_sdf.append(loss_sdf.item())
        training_losses_trav.append(loss_trav.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_epoch_loss}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                locations = batch['locations'].to(device)
                point_clouds = batch['point_clouds'].to(device)
                gt_sdf = batch['gt_sdf'].to(device)
                gt_confidence = batch['gt_confidence'].to(device)
                gt_semantics = batch['gt_semantics'].to(device)
                gt_color = batch['gt_color'].to(device)
                audio_data = batch['mel_spectrograms'].to(device)
                traversability = batch['gt_travs'].to(device)

                if len(point_clouds.shape) == 4:
                    batch_size, scans_per_batch, channels, num_points = point_clouds.shape
                    point_clouds = point_clouds.view(batch_size * scans_per_batch, channels, num_points)

                batch_size, scans_per_batch, channels, height, width = audio_data.shape
                audio_data = audio_data.view(batch_size * scans_per_batch, channels, height, width)

                # Reshape locations if necessary
                if len(locations.shape) == 2:
                    num_locations = locations.shape[0] // batch_size
                    locations = locations.view(batch_size, num_locations, 3)

                preds_sdf, preds_confidence, preds_semantics, preds_color_logits, preds_trav = model(locations, point_clouds, audio_data)
                preds_sdf = preds_sdf.view(-1)
                preds_confidence = preds_confidence.view(-1)
                preds_trav = preds_trav.view(batch_size, -1)
                gt_sdf = gt_sdf.view(-1)
                gt_confidence = gt_confidence.view(-1)
                preds_trav_mean = preds_trav.mean(dim=1)
                traversability_mean = traversability.mean(dim=1)

                # Loss calculations
                loss_sdf_val = cfg.WEIGHT_SDF * criterion_huber(preds_sdf, gt_sdf)
                loss_confidence_val = cfg.WEIGHT_CONFIDENCE * criterion_l_conf(preds_confidence, gt_confidence)
                loss_semantics_val = cfg.WEIGHT_SEMANTICS * criterion_ce_semantics(preds_semantics.view(-1, cfg.CLASSES), gt_semantics.long().view(-1))
                preds_color_logits = preds_color_logits.view(-1, cfg.NUM_BINS)
                gt_color = gt_color.view(-1)
                loss_color_val = cfg.WEIGHT_COLOR * criterion_ce_color(preds_color_logits, gt_color)
                loss_trav_val = cfg.WEIGHT_TRAV * criterion_mse(preds_trav_mean, traversability_mean)

                val_loss += loss_sdf_val + loss_confidence_val + loss_semantics_val + loss_color_val + loss_trav_val

        average_val_loss = val_loss / len(val_dataloader)
        sdf_val_loss = loss_sdf_val.item()

        validation_losses.append(average_val_loss.item())
        validation_losses_semantics.append(loss_semantics_val.item())
        validation_losses_color.append(loss_color_val.item())
        validation_losses_confidence.append(loss_confidence_val.item())
        validation_losses_sdf.append(sdf_val_loss)
        validation_losses_trav.append(loss_trav_val.item())

        print(f"Validation Loss: {average_val_loss.item()}")

        if sdf_val_loss < best_sdf_val_loss:
            best_sdf_val_loss = sdf_val_loss
            epochs_no_improve_sdf = 0
        else:
            epochs_no_improve_sdf += 1

        if (epochs_no_improve_sdf >= cfg.EARLY_STOP_EPOCHS) and (epoch >= 200):
            print(f"Early stopping triggered at epoch {epoch + 1}. SDF validation loss did not improve for {cfg.EARLY_STOP_EPOCHS} consecutive epochs.")
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at early stopping point with validation loss: {best_sdf_val_loss}")
            break

        if average_val_loss < best_loss:
            best_loss = average_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_loss}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': average_epoch_loss,
            'best_loss': best_loss
        }, checkpoint_path)

        scheduler.step(average_val_loss)

    # End timing and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time // 3600:.0f} hours, {(total_time % 3600) // 60:.0f} minutes, {total_time % 60:.0f} seconds")

    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model
model = MultiModalNetwork()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model = model.to(device)

# Load data
train_file = np.load(cfg.TRAIN_FILE_PATH, allow_pickle=True)
val_file = np.load(cfg.VAL_FILE_PATH, allow_pickle=True)

train_dataset = CustomDataset(preloaded_data=train_file, num_bins=cfg.NUM_BINS, points_per_scan=cfg.POINTS_PER_SCAN)
val_dataset = CustomDataset(preloaded_data=val_file, num_bins=cfg.NUM_BINS, points_per_scan=cfg.POINTS_PER_SCAN)

train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn, drop_last=True)

# Train and validate the model
trained_model = train_val(
    model,
    train_dataloader,
    val_dataloader,
    epochs=cfg.EPOCHS,
    lr=cfg.LR,
    checkpoint_path=cfg.CHECKPOINT_PATH,
    best_model_path=cfg.BEST_MODEL_PATH,
    use_pretrained=not args.scratch )
