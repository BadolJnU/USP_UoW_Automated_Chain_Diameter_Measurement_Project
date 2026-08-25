import json
import os
from PIL import Image

# Path settings
json_path = "C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/Data/ground_truth.json"
train_img_dir = "C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/Data/Images/Train"

with open(json_path, "r") as f:
    data = json.load(f)

print(data)

# Loop through images in your JSON and convert annotations
# Example structure assuming standard COCO-style JSON format:
for img_info in data.get("images", []):
    img_id = img_info["id"]
    file_name = img_info["file_name"]
    img_width = img_info["width"]
    img_height = img_info["height"]

    txt_filename = os.path.splitext(file_name)[0] + ".txt"
    txt_path = os.path.join(train_img_dir, txt_filename)

    with open(txt_path, "w") as txt_file:
        for ann in data.get("annotations", []):
            if ann["image_id"] == img_id:
                # Get COCO bbox [x_min, y_min, width, height]
                x_min, y_min, w, h = ann["bbox"]

                # Convert to YOLO format (normalized center_x, center_y, w, h)
                x_center = (x_min + w / 2) / img_width
                y_center = (y_min + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height

                class_id = ann.get("category_id", 0)
                txt_file.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                )