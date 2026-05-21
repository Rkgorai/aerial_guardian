"""
Convert VisDrone MOT format to YOLO format for validation set.

VisDrone MOT annotation format:
frame_id, track_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

YOLO format (per image):
class_id x_center y_center width height (all normalized 0-1)

Target classes for this assignment:
- 1: pedestrian (mapped to YOLO class 0)
- 2: people (mapped to YOLO class 0 - both are "persons")
"""

import os
import shutil
from pathlib import Path


def convert_visdrone_to_yolo(
    visdrone_root: str,
    output_root: str,
    target_classes: list = [1, 2],
    split: str = "val",
):
    """
    Convert VisDrone MOT annotations to YOLO format with flattened directory structure.

    Args:
        visdrone_root: Path to VisDrone2019-MOT-val directory
        output_root: Path to output YOLO dataset directory
        target_classes: List of VisDrone category IDs to include (1=pedestrian, 2=people)
        split: Dataset split name (val)
    """
    visdrone_root = Path(visdrone_root)
    output_root = Path(output_root)

    # Create output directories (No sequence subdirectories will be made here)
    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    annotations_dir = visdrone_root / "annotations"
    sequences_dir = visdrone_root / "sequences"

    # Get all annotation files
    annotation_files = list(annotations_dir.glob("*.txt"))
    print(f"Found {len(annotation_files)} annotation files")

    total_frames = 0
    total_persons = 0
    skipped_frames = 0

    for ann_file in annotation_files:
        sequence_name = ann_file.stem
        print(f"\nProcessing: {sequence_name}")

        # Read annotations
        with open(ann_file, "r") as f:
            lines = f.readlines()

        # Group annotations by frame_id
        frame_annotations = {}
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue

            frame_id = int(parts[0])
            track_id = int(parts[1])
            bbox_left = float(parts[2])
            bbox_top = float(parts[3])
            bbox_width = float(parts[4])
            bbox_height = float(parts[5])
            score = float(parts[6])
            object_category = int(parts[7])
            truncation = float(parts[8])
            occlusion = float(parts[9])

            # Filter for target classes (persons only)
            if object_category not in target_classes:
                continue

            # Skip heavily truncated or occluded objects (optional - can adjust threshold)
            if truncation > 0.8 or occlusion > 0.8:
                continue

            if frame_id not in frame_annotations:
                frame_annotations[frame_id] = []

            frame_annotations[frame_id].append(
                {
                    "class_id": 0,  # All persons map to class 0
                    "bbox_left": bbox_left,
                    "bbox_top": bbox_top,
                    "bbox_width": bbox_width,
                    "bbox_height": bbox_height,
                }
            )

        # Get image dimensions from first image in sequence
        seq_dir = sequences_dir / sequence_name
        first_image = list(seq_dir.glob("*.jpg"))[0]
        from PIL import Image

        with Image.open(first_image) as img:
            img_width, img_height = img.size

        print(f"  Image size: {img_width}x{img_height}")
        print(f"  Frames with annotations: {len(frame_annotations)}")

        # Process each frame
        for frame_id, annotations in sorted(frame_annotations.items()):
            frame_num = f"{frame_id:07d}"
            image_file = seq_dir / f"{frame_num}.jpg"
            
            # --- NEW NAMING CONVENTION ---
            # Format: sequenceName_frameNumber.jpg/txt
            new_filename_base = f"{sequence_name}_{frame_num}"
            label_file = labels_dir / f"{new_filename_base}.txt"

            if not image_file.exists():
                print(f"  Warning: Image not found {image_file}")
                skipped_frames += 1
                continue

            # Copy image directly to the split directory with the new name
            shutil.copy(image_file, images_dir / f"{new_filename_base}.jpg")

            # Convert to YOLO format and save
            yolo_annotations = []
            for ann in annotations:
                # Convert from (x_left, y_top, width, height) to (x_center, y_center, width, height)
                x_center = (ann["bbox_left"] + ann["bbox_width"] / 2) / img_width
                y_center = (ann["bbox_top"] + ann["bbox_height"] / 2) / img_height
                width = ann["bbox_width"] / img_width
                height = ann["bbox_height"] / img_height

                # Clamp values to [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                yolo_annotations.append(
                    f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
                total_persons += 1

            # Save YOLO format annotation in the flat directory
            with open(label_file, "w") as f:
                f.write("\n".join(yolo_annotations))

            total_frames += 1

        print(f"  Processed {len(frame_annotations)} frames")

    print(f"\n{'='*50}")
    print(f"Conversion Complete!")
    print(f"Total frames: {total_frames}")
    print(f"Total person annotations: {total_persons}")
    print(f"Skipped frames: {skipped_frames}")
    print(f"Output directory: {output_root}")
    print(f"{'='*50}")

    # Create dataset.yaml for YOLO training
    dataset_yaml = output_root / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {output_root.absolute()}\n")
        f.write(f"{split}: images/{split}\n\n")
        f.write("names:\n")
        f.write("  0: person\n")

    print(f"\nCreated dataset.yaml at {dataset_yaml}")


if __name__ == "__main__":
    visdrone_root = "VisDrone2019-MOT-val"
    output_root = "yolo_dataset"

    convert_visdrone_to_yolo(
        visdrone_root=visdrone_root,
        output_root=output_root,
        target_classes=[1, 2],  # pedestrian and people
        split="val",
    )