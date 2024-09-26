import numpy as np
import os
import argparse

def extract_numerical_part(filename):
    return ''.join(filter(str.isdigit, filename))

def process_and_save_ground_truth(data_path, output_dir=None, confidence_dir=None):
    # Define the default directories if none are provided
    base_dir = os.path.join(os.getenv("HOME"), "WildFusion")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "ground_truth")
    if confidence_dir is None:
        confidence_dir = os.path.join(output_dir, "confidence")

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(confidence_dir, exist_ok=True)

    # Load the data
    data = np.load(data_path)

    # Extract the relevant arrays
    locations = data['sampled_points']
    gt_sdfs = data['gt_sdfs'].flatten()  # Flatten the gt_sdfs array
    colors = data['gt_colors']
    semantics = data['gt_semantics'].flatten()  # Flatten the semantics array
    confidences = data['gt_confidences'].flatten()  # Flatten the confidences array
    traversability = data['gt_trav']  # One traversability score per scene
    scan_indices = data['scan_indices']  # Used to distinguish between scenes

    # Debugging statements to print shapes and first few values
    print(f"Processing {data_path}")
    print("Shapes before flattening:")
    print(f"locations: {locations.shape}")
    print(f"gt_sdfs: {gt_sdfs.shape}")
    print(f"colors: {colors.shape}")
    print(f"semantics: {semantics.shape}")
    print(f"confidences: {confidences.shape}")
    print(f"traversability: {traversability.shape}")
    print(f"scan_indices: {scan_indices.shape}")
    print("First few traversability values:", traversability[:4])

    # Iterate through each unique scene
    unique_scenes = np.unique(scan_indices)
    for scene_id in unique_scenes:
        scene_mask = (scan_indices == scene_id)

        # Apply filtering for each scene based on gt_sdf <= 0 and semantics != 0
        mask = (gt_sdfs <= 0) & (semantics != 0) & scene_mask
        filtered_locations = locations[mask]
        filtered_colors = colors[mask]
        filtered_semantics = semantics[mask]
        filtered_sdfs = gt_sdfs[mask]
        filtered_confidences = confidences[mask]

        # Assign the corresponding traversability score for the entire scene
        scene_traversability = traversability[scene_id]  # Get traversability for this scene
        filtered_traversability = np.full(filtered_sdfs.shape, scene_traversability)

        # Apply weights to traversability based on semantic IDs
        semantic_weights = get_semantic_weights()
        semantic_weight = np.vectorize(semantic_weights.get)(filtered_semantics)

        # Apply distance-based Gaussian weighting
        distance_weight = compute_distance_weight(filtered_locations)

        # Combine both weights (semantic and distance-based) to get the final weight
        combined_weight = semantic_weight * distance_weight

        # Generate filenames based on scene_id
        output_filename = os.path.join(output_dir, f"filtered_gt_scene_{scene_id}.pcd")
        confidence_filename = os.path.join(confidence_dir, f"confidence_gt_scene_{scene_id}.txt")

        # Save the filtered point cloud to a PCD file with an additional weight layer
        save_point_cloud_to_pcd(filtered_locations, filtered_colors, filtered_semantics, filtered_sdfs, filtered_traversability, combined_weight, output_filename)

        # Save confidence scores to a txt file
        save_confidence_scores_to_txt(filtered_locations, filtered_confidences, confidence_filename)

        print(f"Saved filtered ground truth point cloud to {output_filename} and confidence scores to {confidence_filename}")

def get_semantic_weights():
    # Define weights for the semantic IDs based on the provided information
    semantic_weights = {
        0: 0.0,   # VOID
        1: 0.9,   # Gravel
        2: 0.0,   # Construction
        3: 1.0,   # Rock
        4: 0.5,   # Leaves
        5: 0.5,   # HighVeg
        6: 0.0,   # Tree
        7: 0.6,   # Terrain
        8: 0.7,   # Grass
        9: 0.0,   # Branch
        10: 0.9   # HardSoil
    }
    return semantic_weights

def compute_distance_weight(locations, mean=0.0, std_dev=6.0):
    # Compute the Euclidean distance from the origin (0,0,0)
    distances = np.linalg.norm(locations, axis=1)
    
    # Apply Gaussian distribution with the specified mean and standard deviation
    distance_weight = np.exp(-((distances - mean) ** 2) / (2 * std_dev ** 2))
    
    return distance_weight

def save_point_cloud_to_pcd(locations, colors, semantics, sdfs, traversability, weight, filename):
    num_points = locations.shape[0]
    
    with open(filename, 'w') as pcd_file:
        # PCD header
        pcd_file.write("VERSION .7\n")
        pcd_file.write("FIELDS x y z rgb semantic sdf traversability weight\n")
        pcd_file.write("SIZE 4 4 4 4 4 4 4 4\n")
        pcd_file.write("TYPE F F F U I F F F\n")
        pcd_file.write("COUNT 1 1 1 1 1 1 1 1\n")
        pcd_file.write(f"WIDTH {num_points}\n")
        pcd_file.write("HEIGHT 1\n")
        pcd_file.write(f"POINTS {num_points}\n")
        pcd_file.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        pcd_file.write("DATA ascii\n")
        
        # PCD body
        for i in range(num_points):
            rgb = (int(colors[i, 0] * 255) << 16) | (int(colors[i, 1] * 255) << 8) | int(colors[i, 2] * 255)
            pcd_file.write(f"{locations[i, 0]} {locations[i, 1]} {locations[i, 2]} {rgb} {int(semantics[i])} {sdfs[i]} {traversability[i]} {weight[i]}\n")

def save_confidence_scores_to_txt(locations, confidences, filename):
    num_points = locations.shape[0]
    
    # Create a file to save the confidence values
    with open(filename, 'w') as file:
        # Write the data in the format: x, y, z, confidence
        for i in range(num_points):
            file.write(f"{locations[i, 0]} {locations[i, 1]} {locations[i, 2]} {confidences[i]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save ground truth point cloud data.")
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getenv("HOME"), "WildFusion", "Dataset", "val_data.npz"), 
                        help="Path to the input .npz file (default: ~/WildFusion/Dataset/val_data.npz)")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Directory to save the output point cloud files (default: ~/WildFusion/ground_truth/pcd)")
    parser.add_argument("--confidence_dir", type=str, default=None, 
                        help="Directory to save the confidence score files (default: ~/WildFusion/ground_truth/confidence)")

    args = parser.parse_args()

    # Process and save the ground truth based on scenes
    process_and_save_ground_truth(args.data_path, args.output_dir, args.confidence_dir)
