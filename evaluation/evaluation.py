import os
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from scipy.spatial.distance import directed_hausdorff

# Function to extract numerical part from a filename
def extract_numerical_part(filename):
    return ''.join(filter(str.isdigit, filename))

def find_matching_file_pairs(ground_truth_dir, prediction_dir):
    ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.pcd')]
    prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.pcd')]

    file_pairs = []
    gt_numbers = {extract_numerical_part(gt_file) for gt_file in ground_truth_files}
    pred_numbers = {extract_numerical_part(pred_file) for pred_file in prediction_files}

    # Find intersection of numerical parts between ground truth and prediction files
    matching_numbers = gt_numbers.intersection(pred_numbers)

    for num in matching_numbers:
        gt_file = next(f for f in ground_truth_files if extract_numerical_part(f) == num)
        pred_file = next(f for f in prediction_files if extract_numerical_part(f) == num)
        gt_full_path = os.path.join(ground_truth_dir, gt_file)
        pred_full_path = os.path.join(prediction_dir, pred_file)
        file_pairs.append((gt_full_path, pred_full_path))

    return file_pairs


# Function to load PCD data
def load_pcd_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find where data starts
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "DATA ascii":
            data_start = i + 1
            break

    # Load data
    data = np.loadtxt(lines[data_start:], delimiter=' ')
    return data

# Color Metrics (MSE, MAE, PSNR)
def compute_color_metrics(rgb1, rgb2):
    mse = np.mean((rgb1 - rgb2) ** 2)
    mae = np.mean(np.abs(rgb1 - rgb2))
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    return mse, mae, psnr

# SDF Metrics (Hausdorff Distance, Chamfer Distance)
def compute_sdf_metrics(points1, points2):
    hausdorff_dist = max(directed_hausdorff(points1, points2)[0], directed_hausdorff(points2, points1)[0])
    kdtree1 = cKDTree(points1)
    kdtree2 = cKDTree(points2)
    dist1, _ = kdtree1.query(points2, k=1)
    dist2, _ = kdtree2.query(points1, k=1)
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return hausdorff_dist, chamfer_dist

def compute_semantic_metrics(semantics1, semantics2, num_classes=11):
    accuracy = accuracy_score(semantics1, semantics2)
    precision = precision_score(semantics1, semantics2, average='weighted', zero_division=0)
    recall = recall_score(semantics1, semantics2, average='weighted', zero_division=0)
    f1 = f1_score(semantics1, semantics2, average='weighted', zero_division=0)
    iou = jaccard_score(semantics1, semantics2, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1, iou

# Align two point clouds by finding the closest matching points
def align_point_clouds(points1, points2):
    kdtree1 = cKDTree(points1)
    dist, idx = kdtree1.query(points2, k=1)
    aligned_points1 = points1[idx]
    return aligned_points1, idx

def compute_average_metrics(file_pairs, num_classes=11):
    color_mse_sum, color_mae_sum, color_psnr_sum = 0, 0, 0
    sdf_hausdorff_sum, sdf_chamfer_sum = 0, 0
    accuracy_sum, precision_sum, recall_sum, f1_sum, iou_sum = 0, 0, 0, 0, 0

    num_pairs = len(file_pairs)

    for file_path1, file_path2 in file_pairs:
        data1 = load_pcd_data(file_path1)
        data2 = load_pcd_data(file_path2)

        points1 = data1[:, :3]
        rgb1 = data1[:, 3].astype(int)
        sdf1 = data1[:, 5]
        semantic1 = data1[:, 4].astype(int)

        points2 = data2[:, :3]
        rgb2 = data2[:, 3].astype(int)
        sdf2 = data2[:, 5]
        semantic2 = data2[:, 4].astype(int)

        aligned_points1, idx = align_point_clouds(points1, points2)

        r1 = ((rgb1 >> 16) & 255) / 255.0
        g1 = ((rgb1 >> 8) & 255) / 255.0
        b1 = (rgb1 & 255) / 255.0
        rgb_colors1 = np.column_stack((r1, g1, b1))

        r2 = ((rgb2 >> 16) & 255) / 255.0
        g2 = ((rgb2 >> 8) & 255) / 255.0
        b2 = (rgb2 & 255) / 255.0
        rgb_colors2 = np.column_stack((r2, g2, b2))

        rgb_colors1 = rgb_colors1[idx]
        rgb_colors2 = rgb_colors2

        mse_color, mae_color, psnr_color = compute_color_metrics(rgb_colors1, rgb_colors2)
        color_mse_sum += mse_color
        color_mae_sum += mae_color
        color_psnr_sum += psnr_color

        hausdorff_dist, chamfer_dist = compute_sdf_metrics(aligned_points1, points2)
        sdf_hausdorff_sum += hausdorff_dist
        sdf_chamfer_sum += chamfer_dist

        semantic1_aligned = semantic1[idx]
        semantic2_aligned = semantic2

        accuracy, precision, recall, f1, iou = compute_semantic_metrics(semantic1_aligned, semantic2_aligned, num_classes)
        accuracy_sum += accuracy
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
        iou_sum += iou

    avg_mse_color = color_mse_sum / num_pairs
    avg_mae_color = color_mae_sum / num_pairs
    avg_psnr_color = color_psnr_sum / num_pairs
    avg_hausdorff_sdf = sdf_hausdorff_sum / num_pairs
    avg_chamfer_sdf = sdf_chamfer_sum / num_pairs
    avg_accuracy = accuracy_sum / num_pairs
    avg_precision = precision_sum / num_pairs
    avg_recall = recall_sum / num_pairs
    avg_f1 = f1_sum / num_pairs
    avg_iou = iou_sum / num_pairs

    return {
        "color_mse": avg_mse_color,
        "color_mae": avg_mae_color,
        "color_psnr": avg_psnr_color,
        "sdf_hausdorff": avg_hausdorff_sdf,
        "sdf_chamfer": avg_chamfer_sdf,
        "semantic_accuracy": avg_accuracy,
        "semantic_precision": avg_precision,
        "semantic_recall": avg_recall,
        "semantic_f1": avg_f1,
        "semantic_iou": avg_iou
    }

if __name__ == "__main__":
    ground_truth_dir = "folder_path1"
    prediction_dir = "folder_path2"

    # Automatically find matching file pairs based on the numerical part of filenames
    file_pairs = find_matching_file_pairs(ground_truth_dir, prediction_dir)

    # Compute average metrics across all file pairs
    average_metrics = compute_average_metrics(file_pairs)

    # Print the averaged results
    print("Averaged Metrics:")
    for metric, value in average_metrics.items():
        print(f"{metric}: {value}")
