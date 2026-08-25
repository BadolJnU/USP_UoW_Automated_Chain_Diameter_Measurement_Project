import cv2
import numpy as np
from ultralytics import YOLO

def measure_link_dimensions(binary_mask: np.ndarray, scale_cm_per_px: float) -> dict:
    """
    Computes Wire Diameter, Outer Length (A2), and Outer Width (B2) in centimeters.
    
    :param binary_mask: Single link mask (uint8, values 0 or 255)
    :param scale_cm_per_px: Calibration factor converting pixels to centimeters
    :return: Dictionary containing diameter, A2, and B2 in cm
    """
    # 1. Clean mask noise
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    _, clean_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # --- Measure Outer Dimensions: A2 (Length) and B2 (Width) ---
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"wire_diam_cm": 0.0, "A2_cm": 0.0, "B2_cm": 0.0}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # Minimum Area Bounding Box (handles rotated chain links correctly)
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (dim1, dim2), angle = rect
    
    # A2 is the larger dimension (Length), B2 is the smaller dimension (Width)
    length_px = max(dim1, dim2)
    width_px = min(dim1, dim2)
    
    a2_cm = length_px * scale_cm_per_px
    b2_cm = width_px * scale_cm_per_px
    
    # --- Measure Wire Diameter using Distance Transform ---
    dist_map = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    
    # Skeletonization to find centerline
    try:
        skeleton = cv2.ximgproc.thinning(clean_mask)
    except AttributeError:
        # Fallback if opencv-contrib-python is missing
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros(clean_mask.shape, dtype=np.uint8)
        temp_mask = clean_mask.copy()
        while True:
            eroded = cv2.erode(temp_mask, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_mask, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_mask = eroded.copy()
            if cv2.countNonZero(temp_mask) == 0:
                break

    # Sample radii along centerline
    skeleton_radii = dist_map[skeleton > 0]
    if len(skeleton_radii) > 0:
        # Filter end-cap outliers (10th to 90th percentile)
        diameters_px = skeleton_radii * 2.0
        q10, q90 = np.percentile(diameters_px, [10, 90])
        valid_px = diameters_px[(diameters_px >= q10) & (diameters_px <= q90)]
        wire_diam_cm = np.median(valid_px) * scale_cm_per_px
    else:
        wire_diam_cm = 0.0
        
    return {
        "wire_diam_cm": round(float(wire_diam_cm), 3),
        "A2_cm": round(float(a2_cm), 3),
        "B2_cm": round(float(b2_cm), 3)
    }

# --- Execution ---

# 1. Load trained model
model = YOLO("C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/runs/segment/chain_project/link_detector-3/weights/best.pt")

# 2. Calibration Scale (Set your real-world calibration factor in cm per pixel)
SCALE_CM_PER_PX = 0.0082  # e.g., 0.0082 cm per pixel (0.082 mm/px)

image_path = r"C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/Data/Images/Train/000038.jpg"
image = cv2.imread(image_path)
h, w, _ = image.shape

# Predict masks
results = model(image_path, conf=0.15)[0]

if results.masks is not None:
    for i, mask in enumerate(results.masks.data):
        # Resize mask to image dimensions
        single_mask = (mask.cpu().numpy() * 255).astype(np.uint8)
        single_mask = cv2.resize(single_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Compute metrics in cm
        dim = measure_link_dimensions(single_mask, SCALE_CM_PER_PX)
        
        print(f"Link {i+1}:")
        print(f"  • Wire Diameter: {dim['wire_diam_cm']} cm")
        print(f"  • Outer Length (A2): {dim['A2_cm']} cm")
        print(f"  • Outer Width  (B2): {dim['B2_cm']} cm")