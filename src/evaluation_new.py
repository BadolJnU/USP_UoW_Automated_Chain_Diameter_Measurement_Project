#!/usr/bin/env python3
"""
Chain Link Dimension Measurement
Measures A2 (outer length), B2 (outer width), D2 (wire diameter), D1 (stud/central vertical)
"""

import cv2
import numpy as np
from skimage.measure import CircleModel, EllipseModel, ransac
from skimage.filters import gaussian
import argparse
import os


# =============================================================================
# DEBUG HELPER -- saves one stage image into the debug folder
# =============================================================================

def make_debug_dir(image_path):
    base = os.path.splitext(image_path)[0]
    debug_dir = base + "_debug_stages"
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir


def save_stage(debug_dir, stage_num, name, img):
    if debug_dir is None:
        return
    fname = f"{stage_num:02d}_{name}.png"
    cv2.imwrite(os.path.join(debug_dir, fname), img)


def draw_points(base_img, points, color=(0, 255, 255), radius=1):
    vis = base_img.copy()
    for x, y in points:
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
    return vis


def draw_circle_fit(base_img, model, inlier_points, color=(0, 255, 0)):
    vis = base_img.copy()
    if inlier_points is not None:
        vis = draw_points(vis, inlier_points, color=(0, 165, 255), radius=1)  # orange = inliers
    if model is not None:
        cx, cy = int(model.center[0]), int(model.center[1])
        r = int(model.radius)
        cv2.circle(vis, (cx, cy), r, color, 2)
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 15, 2)
    return vis


# =============================================================================
# ORIGINAL FUNCTIONS -- enhance_image fixed with adaptive Canny added
# =============================================================================

def enhance_image(img):
    """CLAHE + bilateral filter for rusty metal"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return enhanced, gray


def get_edge_points(gray, low=None, high=None, adaptive=True):
    """FIX: fixed low=30/high=100 thresholds don't adapt to your photo's
    actual contrast. adaptive=True (default) computes thresholds from
    the image's own median brightness instead, which is far more
    robust across different lighting conditions. Set adaptive=False to
    force the original fixed values for comparison."""
    if adaptive:
        median_val = np.median(gray)
        low = int(max(0, 0.5 * median_val))
        high = int(min(255, 1.3 * median_val))
    else:
        low = low if low is not None else 30
        high = high if high is not None else 100
    edges = cv2.Canny(gray, low, high)
    ys, xs = np.where(edges > 0)
    return np.column_stack((xs, ys)), edges


def fit_circle_ransac(points, residual=2.5, max_trials=800):
    if len(points) < 20:
        return None, 0, None
    if len(points) > 2500:
        idx = np.random.choice(len(points), 2500, replace=False)
        points = points[idx]
    try:
        model, inliers = ransac(
            points, CircleModel,
            min_samples=3,
            residual_threshold=residual,
            max_trials=max_trials
        )
        return model, np.sum(inliers), points[inliers]
    except Exception:
        return None, 0, None


def fit_ellipse_ransac(points, residual=4.0, max_trials=600):
    if len(points) < 40:
        return None, 0
    if len(points) > 2000:
        idx = np.random.choice(len(points), 2000, replace=False)
        points = points[idx]
    try:
        model, inliers = ransac(
            points, EllipseModel,
            min_samples=5,
            residual_threshold=residual,
            max_trials=max_trials
        )
        return model, np.sum(inliers)
    except Exception:
        return None, 0


def estimate_wire_diameter(gray, center, outer_r, search_band=40):
    h, w = gray.shape
    cx, cy = int(center[0]), int(center[1])
    r = int(outer_r)
    thicknesses = []
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ox = int(cx + (r + 5) * np.cos(angle))
        oy = int(cy + (r + 5) * np.sin(angle))
        ix = int(cx + max(5, r - search_band) * np.cos(angle))
        iy = int(cy + max(5, r - search_band) * np.sin(angle))
        if not (0 <= ox < w and 0 <= oy < h and 0 <= ix < w and 0 <= iy < h):
            continue
        num = 40
        xs = np.linspace(ix, ox, num).astype(int)
        ys = np.linspace(iy, oy, num).astype(int)
        profile = gray[ys, xs]
        grad = np.abs(np.diff(profile.astype(float)))
        if grad.max() > 15:
            peak = np.argmax(grad)
            thicknesses.append(peak * (search_band / num))
    if thicknesses:
        return float(np.median(thicknesses))
    return None


# =============================================================================
# MAIN ANALYSIS -- now with a debug image saved at every stage
# =============================================================================

def analyze_link(image_path, pixels_per_cm, debug=True, adaptive_canny=True):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {image_path}")

    debug_dir = make_debug_dir(image_path) if debug else None
    original = img.copy()
    save_stage(debug_dir, 0, "original", original)

    enhanced, gray = enhance_image(img)
    save_stage(debug_dir, 1, "clahe_enhanced_color", enhanced)
    save_stage(debug_dir, 2, "gray_bilateral", gray)

    points, edges = get_edge_points(gray, adaptive=adaptive_canny)
    save_stage(debug_dir, 3, "canny_edges", edges)
    save_stage(debug_dir, 4, "all_edge_points_overlay", draw_points(original, points))

    h, w = gray.shape
    results = {
        "full_link_visible": False,
        "visibility_percent": 0.0,
        "A2_cm": None, "B2_cm": None, "D2_cm": None, "D1_cm": None,
        "notes": []
    }

    # ---------- 1. left/right split ----------
    left_mask = (points[:, 0] < w * 0.55) & (points[:, 1] > h * 0.15) & (points[:, 1] < h * 0.85)
    right_mask = (points[:, 0] > w * 0.45) & (points[:, 1] > h * 0.15) & (points[:, 1] < h * 0.85)
    left_pts = points[left_mask]
    right_pts = points[right_mask]

    # DEBUG: visualize the split itself -- if your link is rotated, this
    # stage will visibly cut across the wrong axis. That alone can
    # explain most of a 40%-accuracy problem.
    split_vis = original.copy()
    split_vis = draw_points(split_vis, left_pts, color=(255, 0, 0))   # blue = left window
    split_vis = draw_points(split_vis, right_pts, color=(0, 0, 255))  # red = right window
    cv2.line(split_vis, (int(w * 0.55), 0), (int(w * 0.55), h), (0, 255, 0), 1)
    cv2.line(split_vis, (int(w * 0.45), 0), (int(w * 0.45), h), (0, 255, 255), 1)
    save_stage(debug_dir, 5, "left_right_point_split", split_vis)

    model_L, inl_L, inliers_L = fit_circle_ransac(left_pts, residual=3.0)
    model_R, inl_R, inliers_R = fit_circle_ransac(right_pts, residual=3.0)

    save_stage(debug_dir, 6, "circle_fit_LEFT", draw_circle_fit(original, model_L, inliers_L))
    save_stage(debug_dir, 7, "circle_fit_RIGHT", draw_circle_fit(original, model_R, inliers_R, color=(255, 0, 255)))

    has_left = model_L is not None and inl_L > 80
    has_right = model_R is not None and inl_R > 80

    model_E, inl_E = fit_ellipse_ransac(points, residual=5.0)
    if debug and model_E is not None:
        ellipse_vis = original.copy()
        xc, yc, a, b, theta = model_E.params
        cv2.ellipse(ellipse_vis, (int(xc), int(yc)), (int(a), int(b)),
                    np.degrees(theta), 0, 360, (0, 255, 0), 2)
        save_stage(debug_dir, 8, "ellipse_fit_fallback", ellipse_vis)

    if has_left and has_right:
        results["full_link_visible"] = True
        results["visibility_percent"] = 100.0
        results["notes"].append("Both end circles detected -> full link assumed")
    elif has_left or has_right:
        results["visibility_percent"] = 55.0
        results["notes"].append("Only one end clearly visible -> partial link")
    else:
        results["visibility_percent"] = 20.0
        results["notes"].append("No clear end circles found -> very partial or bad image")

    if has_left and has_right:
        cL = np.array(model_L.center)
        cR = np.array(model_R.center)
        rL = model_L.radius
        rR = model_R.radius
        dist_centers = np.linalg.norm(cL - cR)
        A2_px = dist_centers + rL + rR
        results["A2_cm"] = A2_px / pixels_per_cm
        B2_px = (rL + rR)
        results["B2_cm"] = B2_px / pixels_per_cm

        dL = estimate_wire_diameter(gray, cL, rL)
        dR = estimate_wire_diameter(gray, cR, rR)
        ds = [d for d in (dL, dR) if d is not None]
        if ds:
            D2_px = float(np.mean(ds))
            results["D2_cm"] = D2_px / pixels_per_cm
        else:
            results["notes"].append("D2 estimated from radius difference (less accurate)")
            results["D2_cm"] = ((rL + rR) / 2 * 0.35) / pixels_per_cm

        results["D1_cm"] = (B2_px * 0.45) / pixels_per_cm
        results["notes"].append("D1 is approximate (central vertical). Improve with better side view.")

    elif model_E is not None and inl_E > 150:
        xc, yc, a, b, theta = model_E.params
        major = max(a, b) * 2
        minor = min(a, b) * 2
        results["A2_cm"] = major / pixels_per_cm
        results["B2_cm"] = minor / pixels_per_cm
        results["notes"].append("Used global ellipse fit (less accurate than two-end method)")
        results["visibility_percent"] = max(results["visibility_percent"], 70.0)

    # ---------- final debug visualization (was stage 5, now stage 9) ----------
    if debug:
        vis = original.copy()
        if has_left:
            cv2.circle(vis, (int(model_L.center[0]), int(model_L.center[1])),
                       int(model_L.radius), (0, 255, 0), 2)
        if has_right:
            cv2.circle(vis, (int(model_R.center[0]), int(model_R.center[1])),
                       int(model_R.radius), (0, 255, 0), 2)
        y0 = 30
        for key in ["A2_cm", "B2_cm", "D2_cm", "D1_cm"]:
            val = results[key]
            txt = f"{key}: {val:.2f} cm" if val is not None else f"{key}: N/A"
            cv2.putText(vis, txt, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y0 += 30
        status = "FULL LINK" if results["full_link_visible"] else f"PARTIAL ({results['visibility_percent']:.0f}%)"
        cv2.putText(vis, status, (20, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        save_stage(debug_dir, 9, "FINAL_result", vis)

        out_path = os.path.splitext(image_path)[0] + "_measured.jpg"
        cv2.imwrite(out_path, vis)
        results["debug_image"] = out_path
        results["debug_stage_folder"] = debug_dir

    return results


# ================== MAIN ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure chain link dimensions (A2, B2, D1, D2)")
    parser.add_argument("image", help="Path to chain link image")
    parser.add_argument("--scale", type=float, required=True,
                         help="Pixels per centimeter (e.g. 45.2). Measure a known object in the same photo.")
    parser.add_argument("--no-debug", action="store_true", help="Do not save debug image / stage folder")
    parser.add_argument("--fixed-canny", action="store_true",
                         help="Use the original fixed Canny thresholds (30/100) instead of "
                              "adaptive -- for comparison only")
    args = parser.parse_args()

    np.random.seed(42)

    res = analyze_link(args.image, pixels_per_cm=args.scale, debug=not args.no_debug,
                        adaptive_canny=not args.fixed_canny)

    print("\n========== CHAIN LINK MEASUREMENT ==========")
    print(f"Full link visible : {res['full_link_visible']}")
    print(f"Visibility        : {res['visibility_percent']:.1f} %")
    print("-" * 45)
    for k in ["A2_cm", "B2_cm", "D2_cm", "D1_cm"]:
        v = res[k]
        print(f"{k:8s} : {v:.3f} cm" if v is not None else f"{k:8s} : N/A")
    print("-" * 45)
    for note in res["notes"]:
        print("*", note)
    if "debug_stage_folder" in res:
        print(f"\nDebug stage images saved in: {res['debug_stage_folder']}")
        print("Open that folder and look through 00-09 in order --")
        print("the first stage that looks wrong is where accuracy is being lost.")