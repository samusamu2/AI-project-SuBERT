import os
import json
from PIL import Image

def yolo_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json: str,
    class_names: list
):
    """
    Convert a YOLO-format dataset into a COCO-style JSON file.
    Args:
      - images_dir: path to folder containing .jpg images
      - labels_dir: path to folder containing .txt labels (same basename as images)
      - output_json: where to write train.json
      - class_names: list of class names, length = #classes
    """

    # Gather all image files
    image_files = [
        f for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Populate "categories"
    for cls_id, cls_name in enumerate(class_names):
        coco_dict["categories"].append({
            "id": cls_id,
            "name": cls_name
        })

    ann_id = 0
    for img_id, img_filename in enumerate(image_files):
        img_path = os.path.join(images_dir, img_filename)
        # Open image to get width/height
        with Image.open(img_path) as img:
            width, height = img.size

        # Add image entry
        coco_dict["images"].append({
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })

        # Read corresponding YOLO label file
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.isfile(label_path):
            continue  # no objects in this image

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue

                cls_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to absolute pixel coords
                x_center = x_center_norm * width
                y_center = y_center_norm * height
                w = w_norm * width
                h = h_norm * height

                # COCO wants [x_min, y_min, width, height]
                x_min = x_center - w / 2
                y_min = y_center - h / 2

                # Ensure bbox is within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                if x_min + w > width:
                    w = width - x_min
                if y_min + h > height:
                    h = height - y_min

                # Create annotation entry
                coco_ann = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0
                }
                coco_dict["annotations"].append(coco_ann)
                ann_id += 1

    # Finally, write out the JSON
    with open(output_json, 'w') as out_file:
        json.dump(coco_dict, out_file, indent=2)

    print(f"COCO JSON saved to {output_json}")
    print(f"  - {len(coco_dict['images'])} images")
    print(f"  - {len(coco_dict['annotations'])} total annotations")
    print(f"  - {len(coco_dict['categories'])} categories")


if __name__ == "__main__":
    # === USER PARAMETERS ===
    images_directory = "/home/default/AI-project/yolo_dataset/images/train"
    labels_directory = "/home/default/AI-project/yolo_dataset/labels/train"
    output_coco_json = "/home/default/AI-project/yolo_dataset/annotations/train.json"

    # Build your 1000 class names however you like; here we assume glyph_0000 … glyph_0999
    class_names = [f"glyph_{i:04d}" for i in range(len(os.listdir("glyph_images")))]

    # Make sure output folder exists
    os.makedirs(os.path.dirname(output_coco_json), exist_ok=True)

    # Run conversion
    
    yolo_to_coco(
        images_dir=images_directory,
        labels_dir=labels_directory,
        output_json=output_coco_json,
        class_names=class_names
    )
