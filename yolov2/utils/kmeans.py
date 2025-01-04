import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from sklearn.cluster import KMeans
from yolov2.datasets.voc import VOCDataset


def extract_bounding_boxes(dataset):
    """
    Extract bounding box dimensions (width, height) from a dataset.

    Args:
        dataset (VOCDataset): Dataset object providing bounding boxes.

    Returns:
        np.array: Array of bounding box dimensions (width, height).
    """
    all_boxes = []
    for _, bboxes in dataset:
        for bbox in bboxes:
            width = bbox[2] - bbox[0]  # xmax - xmin
            height = bbox[3] - bbox[1]  # ymax - ymin
            all_boxes.append([width, height])
    return np.array(all_boxes)


def iou(box, clusters):
    """
    Compute IoU (Intersection over Union) between a box and k clusters.

    Args:
        box (array): Single bounding box (width, height).
        clusters (array): Array of cluster centroids (width, height).

    Returns:
        Array of IoUs for the given box with each cluster.
    """
    x_min = np.minimum(box[0], clusters[:, 0])
    y_min = np.minimum(box[1], clusters[:, 1])
    intersection = x_min * y_min
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_values = intersection / (box_area + cluster_area - intersection)
    return iou_values


def avg_iou(boxes, clusters):
    """
    Compute average IoU between a set of boxes and k clusters.

    Args:
        boxes (array): Array of bounding boxes (width, height).
        clusters (array): Array of cluster centroids (width, height).

    Returns:
        Average IoU score.
    """
    return np.mean([np.max(iou(box, clusters)) for box in boxes])


def compute_anchors(boxes, k, max_iter=100):
    """
    Run k-means clustering on bounding boxes to compute anchor boxes.

    Args:
        boxes (array): Array of bounding boxes (width, height).
        k (int): Number of anchor boxes (clusters).
        max_iter (int): Maximum iterations for the k-means algorithm.

    Returns:
        Array of k anchor boxes (width, height).
    """
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, random_state=0).fit(boxes)
    clusters = kmeans.cluster_centers_
    avg_iou_score = avg_iou(boxes, clusters)
    print(f"Average IoU with {k} clusters: {avg_iou_score:.4f}")
    return clusters


if __name__ == "__main__":
    # Initialize VOC dataset
    root_path = "./VOCdevkit/VOC2007"
    dataset = VOCDataset(root=root_path, image_set="trainval", input_size=416)

    # Extract bounding boxes (width, height)
    print("Extracting bounding box dimensions...")
    bounding_boxes = extract_bounding_boxes(dataset)
    print(f"Extracted {len(bounding_boxes)} bounding boxes.")

    # Compute anchor boxes using k-means
    num_clusters = 5
    print(f"Computing {num_clusters} anchor boxes...")
    anchors = compute_anchors(bounding_boxes, num_clusters)
    print(f"Computed Anchors:\n{anchors}")
