import os
import subprocess
from pathlib import Path
from tqdm import tqdm

BEST_WEIGHT_PATH = "<path to best model weights file>"
TEST_VID_DIR = "<test video directory path, containing various video files>"
OUTPUT_VID_DIR = "<output directory for multiple output videos>"
os.makedirs(OUTPUT_VID_DIR, exist_ok=True)

VIDEO_SCRIPT = str(Path(__file__).resolve().parent / "video_visualization.py")

for vid_file in tqdm(sorted(os.listdir(TEST_VID_DIR)), colour="green"):
    if vid_file.endswith(".mp4"):
        vid_id = Path(vid_file).stem
        input_vid_path = os.path.join(TEST_VID_DIR, vid_file)
        output_vid_path = os.path.join(OUTPUT_VID_DIR, vid_id + ".avi")

        command = [
            "python3",
            VIDEO_SCRIPT,
            "-i",
            input_vid_path,
            "-o",
            output_vid_path,
            "-p",
            BEST_WEIGHT_PATH,
        ]

        subprocess.run(command, check=True)

