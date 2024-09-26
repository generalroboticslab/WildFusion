import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models import MultiModalNetwork
from src.data_loader import CustomDataset, custom_collate_fn
from skimage import color
import time
import pandas as pd
import logging
import argparse
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rgb_to_int(rgb):
    """Convert RGB colors to integer."""
    return (rgb[:, 0].astype(int) << 16) + (rgb[:, 1].astype(int) << 8) + rgb[:, 2].astype(int)

def save_point_cloud_to_pcd(locations, colors, semantics, sdfs, traversability, filename):
    """Save point cloud data to a PCD file."""
    num_points = locations.shape[0]
    df = pd.DataFrame({
        'x': locations[:, 0],
        'y': locations[:, 1],
        'z': locations[:, 2],
        'rgb': rgb_to_int(colors),
        'semantic': semantics,
        'sdf': sdfs,
        'traversability': traversability
    })

    with open(filename, 'w') as file:
        file.write("VERSION .7\n")
        file.write("FIELDS x y z rgb semantic sdf traversability\n")
        file.write("SIZE 4 4 4 4 4 4 4\n")
        file.write("TYPE F F F U I F F\n")
        file.write("COUNT 1 1 1 1 1 1 1\n")
        file.write(f"WIDTH {num_points}\n")
        file.write("HEIGHT 1\n")
        file.write(f"POINTS {num_points}\n")
        file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        file.write("DATA ascii\n")

        for _, row in df.iterrows():
            file.write(f"{row['x']} {row['y']} {row['z']} {row['rgb']} {row['semantic']} {row['sdf']} {row['traversability']}\n")

def denormalize_sdf(normalized_sdf, original_min=-6, original_max=6):
    """Denormalize the SDF values."""
    restored_sdf = (normalized_sdf + 1) / 2
    range_original = original_max - original_min
    restored_sdf = restored_sdf * range_original + original_min
    return restored_sdf

def get_semantic_weights():
    """Return the semantic weights based on classes."""
    return {
        0: 0.0, 1: 0.9, 2: 0.0, 3: 1.0, 4: 0.5, 5: 0.5,
        6: 0.0, 7: 0.6, 8: 0.7, 9: 0.0, 10: 0.9
    }

def compute_distance_weight(locations, mean=0.0, std_dev=6.0):
    """Compute the Gaussian-based distance weighting."""
    distances = np.linalg.norm(locations, axis=1)
    return np.exp(-((distances - mean) ** 2) / (2 * std_dev ** 2))

def save_confidence_scores_to_txt(locations, confidences, filename):
    """Save confidence scores to a text file."""
    df = pd.DataFrame({
        'x': locations[:, 0],
        'y': locations[:, 1],
        'z': locations[:, 2],
        'confidence': confidences
    })
    df.to_csv(filename, sep=' ', index=False, header=False)

def test_model_on_data(model, test_file, device, pcd_output_dir, confidence_output_dir):
    """Test the model on the dataset and save the results."""
    preloaded_data = np.load(test_file)
    
    dataset = CustomDataset(preloaded_data, num_bins=313, points_per_batch=30000, scans_per_batch=1)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    model.eval()
    model.to(device)
    
    semantic_weights = get_semantic_weights()

    start_time = time.time()

    scene_counter = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch is None:
            logging.warning(f"Skipping batch {batch_idx} due to data loading error")
            continue

        try:
            locations = batch['locations'].to(device)
            point_clouds = batch['point_clouds'].to(device)
            mel_spectrogram = batch['mel_spectrograms'].to(device)

            if len(locations.shape) == 2:
                batch_size = point_clouds.shape[0]
                num_locations = locations.shape[0] // batch_size
                locations = locations.view(batch_size, num_locations, 3)

            point_clouds = point_clouds.squeeze(1)

            if len(mel_spectrogram.shape) == 5:
                mel_spectrogram = mel_spectrogram.squeeze(1)
        
            with torch.no_grad():
                preds_sdf, preds_confidence, preds_semantics, preds_color, preds_traversability = model(locations, point_clouds, mel_spectrogram)
        
            preds_sdf = denormalize_sdf(preds_sdf)
            preds_sdf = preds_sdf.cpu().numpy().squeeze()

            mask = preds_sdf <= 0.0
            locations_np = locations.cpu().numpy().reshape(-1, 3)
            locations_np_filtered = locations_np[mask]

            preds_traversability = preds_traversability.cpu().numpy().squeeze()[mask]
            preds_semantics = preds_semantics.cpu().numpy().squeeze()[mask]
            preds_color = preds_color.cpu().numpy()[mask]
            preds_color = np.argmax(preds_color, axis=2)

            preds_confidence = preds_confidence.cpu().numpy().squeeze()[mask]

            if preds_semantics.size > 0:
                semantic_weight = np.vectorize(semantic_weights.get)(np.argmax(preds_semantics, axis=1))
            else:
                semantic_weight = np.array([])

            distance_weight = compute_distance_weight(locations_np_filtered)

            weighted_traversability = preds_traversability * semantic_weight * distance_weight

            num_bins = 313
            lab_colors = preds_color / (num_bins - 1)
            lab_colors = lab_colors * [100, 255, 255] - [0, 128, 128]
            rgb_colors = color.lab2rgb(lab_colors)
            final_colors = (rgb_colors * 255).astype(np.uint8)

            final_semantics = np.argmax(preds_semantics, axis=1)

            valid_mask = final_semantics != 0
            final_locations = locations_np_filtered[valid_mask]
            final_colors = final_colors[valid_mask]
            final_semantics = final_semantics[valid_mask]
            final_sdfs = preds_sdf[mask][valid_mask]
            final_traversability = weighted_traversability[valid_mask]
            final_confidences = preds_confidence[valid_mask]

            pcd_filename = os.path.join(pcd_output_dir, f"predicted_output_{scene_counter}.pcd")
            save_point_cloud_to_pcd(final_locations, final_colors, final_semantics, final_sdfs, final_traversability, pcd_filename)

            confidence_filename = os.path.join(confidence_output_dir, f"confidence_scores_{scene_counter}.txt")
            save_confidence_scores_to_txt(final_locations, final_confidences, confidence_filename)

            scene_counter += 1
            logging.info(f"Processed scene {scene_counter}")

        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
            continue

    end_time = time.time()
    logging.info(f"Time taken to predict all scenes: {end_time - start_time:.2f} seconds")
    logging.info(f"Total number of scenes processed: {scene_counter}")

def main():
    # Define the root directory
    wild_fusion_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description="Test MultiModal Network on dataset")
    parser.add_argument("--model_path", type=str, 
                        default=os.path.join(wild_fusion_root, 'saved_model_test', 'best_model.pth'),
                        help="Path to the best model checkpoint")
    parser.add_argument("--test_file", type=str, 
                        default=os.path.join(wild_fusion_root, 'Dataset', 'test_data.npz'),
                        help="Path to the test data file")
    parser.add_argument("--pcd_output", type=str, 
                        default=os.path.join(wild_fusion_root, 'prediction_results', 'pcd'),
                        help="Directory to save PCD outputs")
    parser.add_argument("--confidence_output", type=str, 
                        default=os.path.join(wild_fusion_root, 'prediction_results', 'confidence'),
                        help="Directory to save confidence scores")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = MultiModalNetwork().to(device)
    state_dict = torch.load(args.model_path)
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    os.makedirs(args.pcd_output, exist_ok=True)
    os.makedirs(args.confidence_output, exist_ok=True)

    test_model_on_data(model, args.test_file, device, args.pcd_output, args.confidence_output)

if __name__ == "__main__":
    main()