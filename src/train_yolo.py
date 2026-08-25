from ultralytics import YOLO

# 1. Load pretrained segmentation weights
model = YOLO("yolov8n-seg.pt")

# 2. Start training on your custom labels
model.train(
    data="C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/Data//data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="chain_project",
    name="link_detector"
)