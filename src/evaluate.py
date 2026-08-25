#!/usr/bin/env python3
"""
Chain Measurement Evaluation Pipeline
=====================================
Compares vision-based measurements against ground truth caliper data.

Usage:
    # Classical detection (no model needed, quick test):
    python evaluate.py --images ../data/images/test --classical

    # YOLO detection (requires trained model):
    python evaluate.py --images ../data/images/test --model runs/segment/chain/weights/best.pt

    # Full options:
    python evaluate.py --images path/to/images --gt ../data/ground_truth.json \
                       --model best.pt --output ../results --conf 0.5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from ultralytics import YOLO

from measure_utils import (
    detect_links_classical,
    detect_links_yolo,
    load_ground_truth,
    compute_scale_factor,
    evaluate_detections,
    draw_results
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate chain measurement accuracy")
    parser.add_argument("--images", required=True, type=Path,
                        help="Folder containing test images (.jpg, .png)")
    parser.add_argument("--gt", type=Path, default=Path("C:/Users/mb1459/OneDrive - The University of Waikato/Documents/Project_Code_Base/version_3/Data/ground_truth.json"),
                        help="Path to ground_truth.json")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to YOLO .pt model (omit to use classical)")
    parser.add_argument("--output", type=Path, default=Path("../Results"),
                        help="Output directory for reports and plots")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO confidence threshold")
    parser.add_argument("--classical", action="store_true",
                        help="Use classical detection instead of YOLO")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process (for quick testing)")
    return parser.parse_args()


def process_image(image_path: Path, model, gt: Dict, use_classical: bool, 
                  conf: float) -> Dict:
    """Process a single image and return evaluation results."""

    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"Cannot load image: {image_path}"}

    # Detect links
    if use_classical or model is None:
        detections = detect_links_classical(img)
    else:
        detections = detect_links_yolo(img, model, conf=conf)

    if len(detections) == 0:
        return {
            "image": str(image_path.name),
            "n_detected": 0,
            "error": "No links detected"
        }

    # Compute scale from GT vs detected dimensions
    scale = compute_scale_factor(detections, gt)
    if scale is None:
        return {
            "image": str(image_path.name),
            "n_detected": len(detections),
            "error": "Could not compute scale factor"
        }

    # Evaluate
    evaluation = evaluate_detections(detections, gt, scale)
    evaluation["image"] = str(image_path.name)
    evaluation["n_detected"] = len(detections)

    # Draw visualization
    viz = draw_results(img, detections, evaluation, gt)

    return {
        "evaluation": evaluation,
        "visualization": viz,
        "raw_detections": len(detections)
    }


def aggregate_across_images(all_results: List[Dict]) -> Dict:
    """Aggregate statistics across all processed images."""

    # Collect all per-link errors
    wire_errors = []
    a2_errors = []
    b2_errors = []
    ratio_errors = []

    wire_abs_pct = []
    a2_abs_pct = []
    b2_abs_pct = []

    image_count = 0
    total_detections = 0

    for res in all_results:
        if "evaluation" not in res or "error" in res["evaluation"]:
            continue

        image_count += 1
        ev = res["evaluation"]
        total_detections += ev.get("n_links_detected", 0)

        for link_id, link_data in ev.get("per_link", {}).items():
            errs = link_data["errors"]

            if errs.get("wire_err_cm") is not None:
                wire_errors.append(errs["wire_err_cm"])
                if errs.get("wire_err_pct") is not None:
                    wire_abs_pct.append(abs(errs["wire_err_pct"]))

            if errs.get("A2_err_cm") is not None:
                a2_errors.append(errs["A2_err_cm"])
                if errs.get("A2_err_pct") is not None:
                    a2_abs_pct.append(abs(errs["A2_err_pct"]))

            if errs.get("B2_err_cm") is not None:
                b2_errors.append(errs["B2_err_cm"])
                if errs.get("B2_err_pct") is not None:
                    b2_abs_pct.append(abs(errs["B2_err_pct"]))

            if errs.get("ratio_err") is not None:
                ratio_errors.append(errs["ratio_err"])

    def _agg(errors, pct_errors):
        arr = np.array(errors)
        return {
            "count": len(arr),
            "MAE_cm": round(float(np.mean(np.abs(arr))), 4),
            "RMSE_cm": round(float(np.sqrt(np.mean(arr**2))), 4),
            "mean_error_cm": round(float(np.mean(arr)), 4),
            "std_error_cm": round(float(np.std(arr)), 4),
            "max_abs_error_cm": round(float(np.max(np.abs(arr))), 4),
            "MAPE_pct": round(float(np.mean(pct_errors)), 2) if pct_errors else None
        }

    summary = {
        "images_processed": image_count,
        "total_link_detections": total_detections,
        "mean_links_per_image": round(total_detections / max(image_count, 1), 2)
    }

    if wire_errors:
        summary["wire_diameter"] = _agg(wire_errors, wire_abs_pct)
    if a2_errors:
        summary["A2_length"] = _agg(a2_errors, a2_abs_pct)
    if b2_errors:
        summary["B2_width"] = _agg(b2_errors, b2_abs_pct)
    if ratio_errors:
        arr = np.array(ratio_errors)
        summary["A2_B2_ratio"] = {
            "MAE": round(float(np.mean(np.abs(arr))), 4),
            "mean_error": round(float(np.mean(arr)), 4),
            "std_error": round(float(np.std(arr)), 4)
        }

    return summary


def generate_plots(all_results: List[Dict], output_dir: Path):
    """Generate accuracy analysis plots."""

    # Extract data for plotting
    wire_preds = {str(i): [] for i in range(1, 5)}
    wire_gts = {str(i): [] for i in range(1, 5)}
    a2_preds = {str(i): [] for i in range(1, 5)}
    b2_preds = {str(i): [] for i in range(1, 5)}
    scales = []

    for res in all_results:
        if "evaluation" not in res:
            continue
        ev = res["evaluation"]
        scales.append(ev.get("scale_cm_per_px", 0))

        for link_id, data in ev.get("per_link", {}).items():
            if "predicted" in data and "ground_truth" in data:
                wire_preds[link_id].append(data["predicted"]["wire_diam_cm"])
                wire_gts[link_id].append(data["ground_truth"]["wire_diam_cm"])
                a2_preds[link_id].append(data["predicted"]["A2_cm"])
                b2_preds[link_id].append(data["predicted"]["B2_cm"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Chain Measurement Accuracy vs Ground Truth", fontsize=14, fontweight='bold')

    # 1. Wire diameter: predicted vs GT per link
    ax = axes[0, 0]
    link_ids = [str(i) for i in range(1, 5)]
    gt_vals = [np.mean(wire_gts[lid]) if wire_gts[lid] else 0 for lid in link_ids]
    pred_means = [np.mean(wire_preds[lid]) if wire_preds[lid] else 0 for lid in link_ids]
    pred_stds = [np.std(wire_preds[lid]) if wire_preds[lid] else 0 for lid in link_ids]

    x = np.arange(len(link_ids))
    width = 0.35
    ax.bar(x - width/2, gt_vals, width, label='Ground Truth', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, pred_means, width, yerr=pred_stds, label='Vision (mean±std)', 
           color='coral', alpha=0.8, capsize=3)
    ax.set_xlabel("Link ID")
    ax.set_ylabel("Wire Diameter (cm)")
    ax.set_title("Wire Diameter Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Link {lid}" for lid in link_ids])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Wire diameter error distribution
    ax = axes[0, 1]
    all_wire_errs = []
    for lid in link_ids:
        for p, g in zip(wire_preds[lid], wire_gts[lid]):
            all_wire_errs.append(p - g)
    if all_wire_errs:
        ax.hist(all_wire_errs, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(np.mean(all_wire_errs), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(all_wire_errs):.3f}cm')
        ax.set_xlabel("Error (cm)")
        ax.set_ylabel("Count")
        ax.set_title("Wire Diameter Error Distribution")
        ax.legend()
        ax.grid(alpha=0.3)

    # 3. Scale factor consistency across images
    ax = axes[0, 2]
    if scales:
        ax.plot(scales, 'o-', color='teal', markersize=4)
        ax.axhline(np.median(scales), color='red', linestyle='--', 
                   label=f'Median: {np.median(scales):.5f}')
        ax.set_xlabel("Image #")
        ax.set_ylabel("Scale (cm/pixel)")
        ax.set_title("Scale Factor Consistency")
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. A2 comparison
    ax = axes[1, 0]
    gt_a2 = [gt_vals_a2 := [np.mean([7.4274,7.2522,7.3714,7.4259])]]  # placeholder
    # Actually compute properly
    a2_gt_vals = []
    a2_pred_vals = []
    for lid in link_ids:
        # Get GT A2 for this link from first result
        for res in all_results:
            if "evaluation" in res and lid in res["evaluation"].get("per_link", {}):
                a2_gt_vals.append(res["evaluation"]["per_link"][lid]["ground_truth"]["A2_cm"])
                a2_pred_vals.append(np.mean(a2_preds[lid]) if a2_preds[lid] else 0)
                break

    if a2_gt_vals and a2_pred_vals:
        x = np.arange(len(link_ids))
        ax.bar(x - width/2, a2_gt_vals, width, label='GT', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, a2_pred_vals, width, label='Vision', color='coral', alpha=0.8)
        ax.set_xlabel("Link ID")
        ax.set_ylabel("A2 Length (cm)")
        ax.set_title("Outer Length (A2) Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Link {lid}" for lid in link_ids])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # 5. B2 comparison
    ax = axes[1, 1]
    b2_gt_vals = []
    b2_pred_vals = []
    for lid in link_ids:
        for res in all_results:
            if "evaluation" in res and lid in res["evaluation"].get("per_link", {}):
                b2_gt_vals.append(res["evaluation"]["per_link"][lid]["ground_truth"]["B2_cm"])
                b2_pred_vals.append(np.mean(b2_preds[lid]) if b2_preds[lid] else 0)
                break

    if b2_gt_vals and b2_pred_vals:
        x = np.arange(len(link_ids))
        ax.bar(x - width/2, b2_gt_vals, width, label='GT', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, b2_pred_vals, width, label='Vision', color='coral', alpha=0.8)
        ax.set_xlabel("Link ID")
        ax.set_ylabel("B2 Width (cm)")
        ax.set_title("Outer Width (B2) Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Link {lid}" for lid in link_ids])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')

    # Build summary text from aggregate
    summary = aggregate_across_images(all_results)
    lines = ["AGGREGATE ACCURACY", "=" * 30]

    if "wire_diameter" in summary:
        wd = summary["wire_diameter"]
        lines.extend([
            f"Wire Diameter:",
            f"  MAE:  {wd['MAE_cm']:.3f} cm",
            f"  RMSE: {wd['RMSE_cm']:.3f} cm",
            f"  MAPE: {wd['MAPE_pct']:.2f}%" if wd['MAPE_pct'] else "  MAPE: N/A",
            f"  Max Error: {wd['max_abs_error_cm']:.3f} cm",
            ""
        ])

    if "A2_length" in summary:
        a2 = summary["A2_length"]
        lines.extend([
            f"A2 Length:",
            f"  MAE: {a2['MAE_cm']:.3f} cm",
            f"  MAPE: {a2['MAPE_pct']:.2f}%" if a2['MAPE_pct'] else "  MAPE: N/A",
            ""
        ])

    if "B2_width" in summary:
        b2 = summary["B2_width"]
        lines.extend([
            f"B2 Width:",
            f"  MAE: {b2['MAE_cm']:.3f} cm",
            f"  MAPE: {b2['MAPE_pct']:.2f}%" if b2['MAPE_pct'] else "  MAPE: N/A",
            ""
        ])

    lines.extend([
        f"Images processed: {summary.get('images_processed', 0)}",
        f"Total detections: {summary.get('total_link_detections', 0)}"
    ])

    ax.text(0.1, 0.5, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Plots] Saved to {output_dir / 'accuracy_plots.png'}")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    if not args.gt.exists():
        print(f"[ERROR] Ground truth not found: {args.gt}")
        print("Make sure ground_truth.json is in the data folder.")
        sys.exit(1)

    gt = load_ground_truth(args.gt)
    print(f"[GT] Loaded measurements for {len(gt['links'])} links")

    # Load YOLO model if specified
    model = None
    if args.model and not args.classical:
        if not args.model.exists():
            print(f"[ERROR] Model not found: {args.model}")
            sys.exit(1)
        print(f"[Model] Loading YOLO from {args.model}")
        model = YOLO(str(args.model))
    elif args.classical:
        print("[Mode] Using classical detection (no YOLO model)")
    else:
        print("[Mode] No model specified, defaulting to classical detection")
        print("        (Use --model path/to/best.pt for YOLO)")
        args.classical = True

    # Find images
    image_paths = sorted(args.images.glob("*.jpg")) + sorted(args.images.glob("*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]

    print(f"[Images] Found {len(image_paths)} images in {args.images}")

    # Process all images
    all_results = []

    for i, img_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] Processing {img_path.name}...")

        result = process_image(img_path, model, gt, args.classical, args.conf)
        all_results.append(result)

        if "error" in result and result["error"]:
            print(f"  -> {result['error']}")
            continue

        ev = result["evaluation"]
        print(f"  -> Detected {ev['n_links_detected']} links, "
              f"scale={ev['scale_cm_per_px']:.5f} cm/px")

        if "aggregate" in ev and "wire_diameter" in ev["aggregate"]:
            wd = ev["aggregate"]["wire_diameter"]
            print(f"  -> Wire MAE this image: {wd['MAE_cm']:.3f} cm")

        # Save visualization
        viz_path = args.output / "annotated" / f"{img_path.stem}_result.jpg"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(viz_path), result["visualization"])

    # Aggregate across all images
    print("\n" + "=" * 60)
    print("AGGREGATE ACCURACY REPORT")
    print("=" * 60)

    summary = aggregate_across_images(all_results)
    print(json.dumps(summary, indent=2))

    # Save full report
    report = {
        "summary": summary,
        "per_image": [
            {k: v for k, v in r.items() if k != "visualization"}
            for r in all_results
        ]
    }

    report_path = args.output / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Full results saved to {report_path}")

    # Generate plots
    if len([r for r in all_results if "evaluation" in r]) > 0:
        generate_plots(all_results, args.output)

    print("=" * 60)
    print("DONE")
    print(f"Annotated images: {args.output / 'annotated'}")
    print(f"JSON report:      {report_path}")
    print(f"Accuracy plots:   {args.output / 'accuracy_plots.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
