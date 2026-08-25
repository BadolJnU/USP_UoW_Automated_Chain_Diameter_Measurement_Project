"""
Chain link measurement pipeline -- A2, B2, pixel coverage, point cloud.
Handles both stud-link and studless chain automatically, and separates
touching/interlocked links using distance-transform + watershed instead
of manual cropping.

CHANGES IN THIS VERSION (debugging over-segmentation):
  - watershed_separate() now takes --peak-frac (default raised from
    0.35 to 0.5) -- higher = fewer, more confident object peaks =
    less fragmentation into tiny false pieces.
  - run_pipeline() now prints the size of EVERY separated region
    before picking the target, so you can see whether watershed found
    a sane number of objects (e.g. 2) or shattered the image into
    many small fragments (e.g. 8+).

INSTALL (once):
    pip install opencv-python numpy

RUN:
    python chain_pipeline.py --image myphoto.jpg --roi 0 340 830 700 \
        --seg canny --peak-frac 0.5 --px-per-cm 16.5 --csv measurements.csv

    --roi X Y W H     : crop region around the link(s). Strongly
                         recommended -- without it, unrelated
                         background/other links in frame can feed
                         false peaks into watershed.
    --seg canny|saturation : segmentation method.
    --peak-frac       : watershed sensitivity, 0-1. Higher = fewer,
                         larger regions (less fragmentation). Try 0.5,
                         then 0.6-0.7 if still fragmented. Print output
                         will show you all region sizes to judge by.
    --px-per-cm       : calibration. Omit for pixel-only output.
    --csv             : results log -- every run APPENDS a new row.
"""

import argparse
import csv
import os
from datetime import datetime
import cv2
import numpy as np


# =============================================================================
# STEP 1-2: load + preprocess
# =============================================================================

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    return img


def preprocess(img_bgr, use_bilateral=True):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return gray


# =============================================================================
# STEP 3: segmentation -- two interchangeable methods
# =============================================================================

def segment_canny(gray_img, sigma_lo=0.5, sigma_hi=1.3):
    median_val = np.median(gray_img)
    lower = int(max(0, sigma_lo * median_val))
    upper = int(min(255, sigma_hi * median_val))
    return cv2.Canny(gray_img, lower, upper)


def segment_saturation(img_bgr, blur_ksize=15):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(sat, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# =============================================================================
# STEP 4: morphological closing
# =============================================================================

def close_gaps(binary_img, kernel_size=15, iterations=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# =============================================================================
# STEP 5-7: fill solid, watershed-separate touching links, pick target
# =============================================================================

def fill_solid(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found -- check --roi and --seg method.")
    solid = np.zeros_like(binary_img)
    cv2.drawContours(solid, contours, -1, 255, -1)
    return solid


def watershed_separate(img_bgr_crop, solid_mask, peak_frac=0.5, dilate_kernel=15):
    dist = cv2.distanceTransform(solid_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, peak_frac * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    sure_bg = cv2.dilate(solid_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(img_bgr_crop.copy(), markers)
    return markers, n_markers  # n_markers = number of separated objects found


def debug_print_region_sizes(markers, n_markers):
    """Prints the pixel area of every separated region so you can judge
    whether watershed found a sane number of objects or fragmented."""
    print(f"\n--- watershed found {max(0, n_markers - 1)} candidate region(s) ---")
    sizes = []
    for label in range(2, n_markers + 2):
        area = int(np.sum(markers == label))
        if area > 0:
            sizes.append((label, area))
    sizes.sort(key=lambda x: -x[1])
    for label, area in sizes:
        print(f"  region label={label:3d}  area={area:8d} px")
    print("--- end region list ---\n")
    return sizes


def select_target_link(markers, n_markers):
    """Pick the largest separated region as the link to measure."""
    best_label, best_area = None, 0
    for label in range(2, n_markers + 2):
        area = int(np.sum(markers == label))
        if area > best_area:
            best_area, best_label = area, label
    if best_label is None:
        raise ValueError("Watershed did not find any separable region.")
    return np.where(markers == best_label, 255, 0).astype(np.uint8)


# =============================================================================
# STEP 8: stud vs studless auto-detection
# =============================================================================

def detect_hole_count(binary_img):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return 0
    hierarchy = hierarchy[0]
    outer_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    min_hole_area = cv2.contourArea(contours[outer_idx]) * 0.01
    n_holes = sum(1 for i, h in enumerate(hierarchy)
                  if h[3] == outer_idx and cv2.contourArea(contours[i]) > min_hole_area)
    return n_holes


# =============================================================================
# STEP 9: rotation-aware stadium fit -> A2, B2
# =============================================================================

def fit_stadium(contour, edge_tol=3, flat_frac=0.7):
    pts = contour.reshape(-1, 2).astype(np.float64)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (cx, cy), (d1, d2), angle = rect
    theta = np.radians(angle if d1 >= d2 else angle + 90)
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    local = (pts - np.array([cx, cy])) @ R.T

    xr = local[:, 0]
    valid = (xr > xr.min() + edge_tol) & (xr < xr.max() - edge_tol)
    lv = local[valid]
    top_pts = lv[lv[:, 1] < 0]
    bot_pts = lv[lv[:, 1] >= 0]
    if len(top_pts) < 10 or len(bot_pts) < 10:
        a2 = max(d1, d2)
        b2 = min(d1, d2)
        return a2, b2, angle

    xmin, xmax = lv[:, 0].min(), lv[:, 0].max()
    m = (1 - flat_frac) / 2
    xlo, xhi = xmin + m * (xmax - xmin), xmax - m * (xmax - xmin)
    top_flat = top_pts[(top_pts[:, 0] > xlo) & (top_pts[:, 0] < xhi)]
    bot_flat = bot_pts[(bot_pts[:, 0] > xlo) & (bot_pts[:, 0] < xhi)]
    if len(top_flat) < 5 or len(bot_flat) < 5:
        a2 = max(d1, d2)
        b2 = min(d1, d2)
        return a2, b2, angle

    B2_px = abs(top_flat[:, 1].mean() - bot_flat[:, 1].mean())
    straight_len = max(top_flat[:, 0].max() - top_flat[:, 0].min(),
                        bot_flat[:, 0].max() - bot_flat[:, 0].min())
    A2_px = straight_len + B2_px
    return A2_px, B2_px, angle


# =============================================================================
# STEP 10: point cloud + pixel coverage
# =============================================================================

def contour_to_point_cloud(contour, sample_every=3):
    pts = contour.reshape(-1, 2)
    return pts[::sample_every]


def draw_point_cloud(img, points_full_coords, radius=2, color=(0, 255, 255)):
    vis = img.copy()
    for p in points_full_coords:
        cv2.circle(vis, tuple(p.astype(int)), radius, color, -1)
    return vis


def save_point_cloud_csv(points_full_coords, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_px", "y_px"])
        for p in points_full_coords:
            writer.writerow([float(p[0]), float(p[1])])


# =============================================================================
# STEP 11: calibration
# =============================================================================

def px_to_cm(px_value, px_per_cm):
    return px_value / px_per_cm


# =============================================================================
# STEP 12: CSV logging (append, auto-increment id)
# =============================================================================

CSV_FIELDS = ["id", "timestamp", "image_path", "link_type", "n_objects_separated",
              "pixel_coverage", "A2_px", "B2_px", "A2_cm", "B2_cm", "angle_deg"]


def append_result_to_csv(csv_path, result, image_path, link_id=None):
    file_exists = os.path.isfile(csv_path)
    if link_id is None:
        if file_exists:
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            numeric = [int(r["id"]) for r in rows if r["id"].isdigit()]
            link_id = (max(numeric) + 1) if numeric else (len(rows) + 1)
        else:
            link_id = 1
    row = {"id": link_id, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "image_path": os.path.abspath(image_path)}
    row.update({k: result.get(k) for k in CSV_FIELDS if k not in row})
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return link_id


# =============================================================================
# ORCHESTRATOR -- runs all steps in order
# =============================================================================

def run_pipeline(image_path, roi=None, seg_method="canny", px_per_cm=None,
                  output_dir=".", point_cloud_sample_every=3, peak_frac=0.5):
    base = os.path.splitext(os.path.basename(image_path))[0]
    img = load_image(image_path)
    h, w = img.shape[:2]

    if roi is None:
        print("WARNING: no --roi given -- using the full image. If watershed "
              "fragments badly below, the first thing to try is adding a "
              "tight --roi around just the link you want to measure.")
        roi = (0, 0, w, h)
    rx, ry, rw, rh = roi
    crop = img[ry:ry + rh, rx:rx + rw]

    # steps 1-2
    gray = preprocess(crop)

    # step 3
    if seg_method == "canny":
        edges = segment_canny(gray)
    elif seg_method == "saturation":
        edges = segment_saturation(crop)
    else:
        raise ValueError("seg_method must be 'canny' or 'saturation'")

    # step 4
    closed = close_gaps(edges)

    # steps 5-7
    solid = fill_solid(closed)
    markers, n_markers = watershed_separate(crop, solid, peak_frac=peak_frac)
    debug_print_region_sizes(markers, n_markers)  # <-- NEW: see all regions
    n_objects = max(0, n_markers - 1)
    link_mask = select_target_link(markers, n_markers)

    link_contours, _ = cv2.findContours(link_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    link_contour = max(link_contours, key=cv2.contourArea)
    pixel_coverage = int(cv2.contourArea(link_contour))

    # step 8
    n_holes = detect_hole_count(link_mask)
    link_type = "stud-link" if n_holes >= 2 else "studless" if n_holes == 1 else "unknown"

    # step 9
    A2_px, B2_px, angle = fit_stadium(link_contour)

    result = {
        "status": "ok", "image": image_path, "link_type": link_type,
        "n_objects_separated": n_objects, "n_holes_detected": n_holes,
        "pixel_coverage": pixel_coverage,
        "A2_px": round(A2_px, 1), "B2_px": round(B2_px, 1), "angle_deg": round(angle, 1),
    }
    if px_per_cm:
        result["A2_cm"] = round(px_to_cm(A2_px, px_per_cm), 2)
        result["B2_cm"] = round(px_to_cm(B2_px, px_per_cm), 2)

    # step 10
    pc_local = contour_to_point_cloud(link_contour, sample_every=point_cloud_sample_every)
    pc_full = pc_local + np.array([rx, ry])
    vis = draw_point_cloud(img, pc_full)
    label = f"{link_type} | A2={A2_px:.0f}px B2={B2_px:.0f}px | pixels={pixel_coverage}"
    if px_per_cm:
        label += f" | A2={result['A2_cm']}cm B2={result['B2_cm']}cm"
    cv2.putText(vis, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, f"{base}_pointcloud.png")
    csv_pc_path = os.path.join(output_dir, f"{base}_pointcloud.csv")
    cv2.imwrite(vis_path, vis)
    save_point_cloud_csv(pc_full, csv_pc_path)
    result["point_cloud_image"] = vis_path
    result["point_cloud_csv"] = csv_pc_path

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--seg", default="canny", choices=["canny", "saturation"])
    ap.add_argument("--peak-frac", type=float, default=0.5,
                     help="watershed sensitivity 0-1, higher = less fragmentation (default 0.5)")
    ap.add_argument("--px-per-cm", type=float, default=None)
    ap.add_argument("--ref-len-cm", type=float, default=None)
    ap.add_argument("--ref-px", type=float, default=None)
    ap.add_argument("--csv", default="measurements.csv")
    ap.add_argument("--id", default=None)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    px_per_cm = args.px_per_cm
    if px_per_cm is None and args.ref_len_cm and args.ref_px:
        px_per_cm = args.ref_px / args.ref_len_cm

    roi = tuple(args.roi) if args.roi else None
    result = run_pipeline(args.image, roi=roi, seg_method=args.seg,
                           px_per_cm=px_per_cm, output_dir=args.output_dir,
                           peak_frac=args.peak_frac)

    print(f"objects separated by watershed : {result['n_objects_separated']}")
    print(f"link type detected             : {result['link_type']}")
    print(f"pixel coverage                 : {result['pixel_coverage']} px^2")
    print(f"A2 = {result['A2_px']} px" + (f"  ({result.get('A2_cm')} cm)" if 'A2_cm' in result else ""))
    print(f"B2 = {result['B2_px']} px" + (f"  ({result.get('B2_cm')} cm)" if 'B2_cm' in result else ""))
    print(f"point cloud image  -> {result['point_cloud_image']}")
    print(f"point cloud points -> {result['point_cloud_csv']}")

    link_id = append_result_to_csv(args.csv, result, args.image, link_id=args.id)
    print(f"appended row id={link_id} to {os.path.abspath(args.csv)}")