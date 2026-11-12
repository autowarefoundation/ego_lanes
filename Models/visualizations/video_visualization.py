import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer, EgoLanesONNXInfer
from Models.visualizations.image_visualization import make_visualization

FRAME_INF_SIZE = (640, 320)
FRAME_ORI_SIZE = (720, 360)


def load_model(checkpoint_path: str):
    if checkpoint_path and checkpoint_path.endswith(".onnx"):
        return EgoLanesONNXInfer(checkpoint_path)
    return EgoLanesNetworkInfer(checkpoint_path=checkpoint_path or "")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--model_checkpoint_path",
        dest="model_checkpoint_path",
        required=True,
        help="Path to PyTorch (.pth) checkpoint or ONNX (.onnx) model file",
    )
    parser.add_argument(
        "-i",
        "--input_video_filepath",
        dest="input_video_filepath",
        required=True,
        help="Path to input video which will be processed by EgoLanes",
    )
    parser.add_argument(
        "-o",
        "--output_video_path",
        dest="output_video_path",
        required=True,
        help="Path to output video where the output video will be saved",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Flag for whether to show frame by frame visualization while processing is occurring",
    )
    args = parser.parse_args()

    model = load_model(args.model_checkpoint_path)
    print("EgoLanes model successfully loaded!")

    print("Reading video")
    cap = cv2.VideoCapture(args.input_video_filepath)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    output_filepath_data = args.output_video_path
    out_dir = os.path.dirname(output_filepath_data)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer_data = cv2.VideoWriter(
        output_filepath_data,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        FRAME_ORI_SIZE,
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(
        range(frame_count),
        desc="Processing video frames: ",
        unit="frames",
        colour="green",
    ):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i} could not be read, skipped.")
            continue

        # Convert OpenCV BGR -> RGB for inference & visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb).resize(FRAME_INF_SIZE)
        image_np = np.array(image)

        prediction = model.inference(image_np)

        vis_image_rgb = np.array(make_visualization(image_np.copy(), prediction))
        vis_image_bgr = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
        vis_image_bgr = cv2.resize(vis_image_bgr, FRAME_ORI_SIZE)

        if args.vis:
            cv2.imshow("EgoLanes Prediction", vis_image_bgr)
            cv2.waitKey(10)

        writer_data.write(vis_image_bgr)

    cap.release()
    writer_data.release()
    cv2.destroyAllWindows()
    print(f"Visualization video saved to: {output_filepath_data}")


if __name__ == "__main__":
    main()

