#!/usr/bin/env python3
"""Benchmark PyTorch and ONNX EgoLanes models on processed validation datasets."""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_ego_lanes import LoadDataEgoLanes, VALID_DATASET_LIST
from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer, EgoLanesONNXInfer

CLASS_NAMES = ["Ego-Left", "Ego-Right", "Other Lanes"]


def compute_channel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def downsample_gt_quarter(mask_hwc: np.ndarray) -> np.ndarray:
    # Match EgoLanesTrainer 1/4 max-pool reduction (320x640 -> 80x160)
    resized = cv2.resize(mask_hwc.astype(np.float32), (640, 320), interpolation=cv2.INTER_NEAREST)
    # 4x4 block max pooling
    pooled = resized.reshape(80, 4, 160, 4, 3).max(axis=(1, 3))
    return pooled > 0.0


def load_validation_samples(dataset_root: str, datasets: list, num_samples_per_dataset: int):
    samples = []
    for ds in datasets:
        ds_dir = os.path.join(dataset_root, ds)
        if not os.path.isdir(ds_dir):
            continue
        loader = LoadDataEgoLanes(
            labels_filepath=os.path.join(ds_dir, "drivable_path.json"),
            mask_dirpath=os.path.join(ds_dir, "mask"),
            images_filepath=os.path.join(ds_dir, "image"),
            dataset=ds,
        )
        _, n_vals = loader.getItemCount()
        count = min(n_vals, num_samples_per_dataset)
        for idx in range(count):
            frame_id, mask_hwc = loader.getItem(idx, is_train=False)
            img_path = os.path.join(ds_dir, "image", f"{frame_id}.jpg")
            img_pil = Image.open(img_path).convert("RGB")
            samples.append(
                {
                    "dataset": ds,
                    "frame_id": frame_id,
                    "image": img_pil,
                    "gt_quarter": downsample_gt_quarter(mask_hwc),
                }
            )
    return samples


def benchmark_model(name: str, model, model_path: str, samples: list, warmup_runs: int):
    if not samples:
        raise ValueError("No validation samples loaded for benchmarking.")

    for i in range(warmup_runs):
        _ = model.inference(samples[i % len(samples)]["image"])

    latencies_ms = []
    ious = [[] for _ in range(3)]

    for sample in samples:
        t0 = time.perf_counter()
        pred_chw = model.inference(sample["image"])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        pred_bool = pred_chw > 0.0
        gt_bool = sample["gt_quarter"]
        for c in range(3):
            ious[c].append(compute_channel_iou(pred_bool[c], gt_bool[:, :, c]))

    per_class_iou = [float(np.mean(c_ious)) for c_ious in ious]
    mean_iou = float(np.mean(per_class_iou))
    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n=== {name} ===")
    print(f"Model size      : {size_mb:.2f} MB")
    print(f"Mean latency    : {np.mean(latencies_ms):.2f} ms ({1000.0 / np.mean(latencies_ms):.1f} FPS)")
    print(f"Mean IoU        : {mean_iou:.4f}")
    for cls_name, cls_iou in zip(CLASS_NAMES, per_class_iou):
        print(f"  {cls_name:<12}: {cls_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch and ONNX EgoLanes models")
    parser.add_argument("--pytorch_model_path", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--fp32_model_path", type=str, default=None, help="Path to FP32 .onnx model")
    parser.add_argument("--int8_model_path", type=str, default=None, help="Path to INT8 .onnx model")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of processed EgoLanes datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CURVELANES", "TUSIMPLE"],
        choices=VALID_DATASET_LIST,
        help="Processed datasets to evaluate on",
    )
    parser.add_argument("--num_samples_per_dataset", type=int, default=50)
    parser.add_argument("--warmup_runs", type=int, default=3)
    args = parser.parse_args()

    samples = load_validation_samples(args.dataset_root, args.datasets, args.num_samples_per_dataset)
    print(f"Loaded {len(samples)} validation samples across {args.datasets}")

    if args.pytorch_model_path:
        pt_model = EgoLanesNetworkInfer(args.pytorch_model_path)
        benchmark_model("PyTorch FP32", pt_model, args.pytorch_model_path, samples, args.warmup_runs)

    if args.fp32_model_path:
        onnx_fp32 = EgoLanesONNXInfer(args.fp32_model_path)
        benchmark_model("ONNX FP32", onnx_fp32, args.fp32_model_path, samples, args.warmup_runs)

    if args.int8_model_path:
        onnx_int8 = EgoLanesONNXInfer(args.int8_model_path)
        benchmark_model("ONNX INT8", onnx_int8, args.int8_model_path, samples, args.warmup_runs)


if __name__ == "__main__":
    main()

