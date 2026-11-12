import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer, EgoLanesONNXInfer

FRAME_INF_SIZE = (640, 320)


def load_model(checkpoint_path: str):
    if checkpoint_path and checkpoint_path.endswith(".onnx"):
        return EgoLanesONNXInfer(checkpoint_path)
    return EgoLanesNetworkInfer(checkpoint_path=checkpoint_path or "")


def make_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    alpha: float = 0.5,
):
    # Compute scale from prediction to image
    img_h, img_w = image.shape[:2]
    _, ph, pw = prediction.shape
    scale_x = img_w / pw
    scale_y = img_h / ph

    # Prep img for drawing (convert to BGR for OpenCV)
    img_bgr = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)

    # Fetch coords (y, x) for each channel
    pred_coords = [np.where(prediction[c] > 0) for c in range(prediction.shape[0])]

    # Choose radius based on scale
    base_scale = min(scale_x, scale_y)
    radius = max(1, int(round(base_scale * 0.5)))

    # Color codes (BGR)
    colors = [
        (255, 153, 0),    # Ego-left lane (Blue)
        (255, 56, 255),   # Ego-right lane (Magenta)
        (87, 255, 87),    # Other lanes (Green)
    ]

    overlay = np.zeros_like(img_bgr)

    # Draw
    for i, (ys, xs) in enumerate(pred_coords):
        if ys.size == 0:
            continue

        xs_scaled = (xs.astype(np.float32) * scale_x).astype(np.int32)
        ys_scaled = (ys.astype(np.float32) * scale_y).astype(np.int32)
        xs_scaled = np.clip(xs_scaled, 0, img_w - 1)
        ys_scaled = np.clip(ys_scaled, 0, img_h - 1)

        color = colors[i]
        for x, y in zip(xs_scaled, ys_scaled):
            x0 = max(0, int(x) - radius)
            x1 = min(img_w - 1, int(x) + radius)
            y0 = max(0, int(y) - radius)
            y1 = min(img_h - 1, int(y) + radius)
            overlay[y0 : y1 + 1, x0 : x1 + 1] = color

    # Blend
    if alpha > 0.0:
        img_bgr = np.clip(
            img_bgr.astype(np.float32) + overlay.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)

    # Back to PIL Image (RGB)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def process_single_image(model, input_image_filepath: str, output_dir: str = None):
    print(f"Reading Image: {input_image_filepath}")
    image = Image.open(input_image_filepath).convert("RGB")
    image = image.resize(FRAME_INF_SIZE)
    image_np = np.array(image)

    prediction = model.inference(image_np)
    vis_image = make_visualization(image_np.copy(), prediction)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_id = Path(input_image_filepath).stem.zfill(3)
        output_image_filepath = os.path.join(output_dir, f"{img_id}_data.png")
        vis_image.save(output_image_filepath)
        print(f"Saved visualization to: {output_image_filepath}")
    else:
        vis_bgr = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
        cv2.imshow("EgoLanes Prediction", vis_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        dest="model_checkpoint_path",
        help="Path to PyTorch (.pth) checkpoint or ONNX (.onnx) model file",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--input_image_dirpath",
        dest="input_image_dirpath",
        help="Path to input image file or directory of images to be processed by EgoLanes",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_image_dirpath",
        dest="output_image_dirpath",
        help="Path to output image directory where visualizations will be saved",
        required=False,
    )
    args = parser.parse_args()

    input_path = args.input_image_dirpath
    output_dir = args.output_image_dirpath

    model_checkpoint_path = args.model_checkpoint_path if args.model_checkpoint_path is not None else ""
    model = load_model(model_checkpoint_path)
    print("EgoLanes model successfully loaded!")

    if os.path.isfile(input_path):
        process_single_image(model, input_path, output_dir)
    elif os.path.isdir(input_path):
        if not output_dir:
            raise ValueError("--output_image_dirpath (-o) is required when input is a directory")
        os.makedirs(output_dir, exist_ok=True)
        for filename in sorted(os.listdir(input_path)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                process_single_image(model, os.path.join(input_path, filename), output_dir)
            else:
                print(f"Skipping non-image file: {filename}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()

