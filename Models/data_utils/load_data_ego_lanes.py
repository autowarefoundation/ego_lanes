#! /usr/bin/env python3

import json
import math
import pathlib
import sys
from typing import Literal, get_args

import cv2
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.check_data import CheckData

VALID_DATASET_LITERALS = Literal[
    "BDD100K",
    "COMMA2K19",
    "CULANE",
    "CURVELANES",
    "ROADWORK",
    "TUSIMPLE",
    "OPENLANE",
    "JIQING",
    "ONCE3DLANE",
]
VAL_SAMPLE_CAPS = {
    "TUSIMPLE": 500,
    "CURVELANES": 1000,
    "JIQING": 1000,
    "CULANE": 1000,
    "BDD100K": 1000,
    "COMMA2K19": 1000,
    "ROADWORK": 1000,
    "OPENLANE": 1000,
    "ONCE3DLANE": 1000,
}
VALID_DATASET_LIST = list(get_args(VALID_DATASET_LITERALS))

FIXED_HOMOTRANS_DATASETS = [
    "CULANE",
    "TUSIMPLE",
]

DYNAMIC_HOMOTRANS_DATASETS = [
    "CURVELANES",
    "OPENLANE",
]


class LoadDataEgoLanes:
    def __init__(
        self,
        labels_filepath: str,
        mask_dirpath: str,
        images_filepath: str,
        dataset: VALID_DATASET_LITERALS,
    ):
        self.labels_filepath = labels_filepath
        self.mask_dirpath = mask_dirpath
        self.image_dirpath = images_filepath
        self.dataset_name = dataset

        if self.dataset_name not in VALID_DATASET_LIST:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'. Valid datasets: {VALID_DATASET_LIST}")

        with open(self.labels_filepath, "r") as f:
            self.labels = json.load(f)

        self.images = sorted(pathlib.Path(self.image_dirpath).glob("*.jpg"))
        self.masks = sorted(pathlib.Path(self.mask_dirpath).glob("*.png"))

        self.N_labels = len(self.labels)
        self.N_masks = len(self.masks)
        self.N_images = len(self.images)

        checkData = CheckData(
            num_images=self.N_images,
            num_gt=self.N_masks,
            dataset_name=self.dataset_name,
        )

        self.train_images = []
        self.train_masks = []
        self.train_ids = []
        self.val_images = []
        self.val_masks = []
        self.val_ids = []

        self.N_trains = 0
        self.N_vals = 0
        self.val_cap = VAL_SAMPLE_CAPS[self.dataset_name]

        if checkData.getCheck():
            for set_idx, frame_id in enumerate(self.labels):
                frame_id_from_img_path = self.images[set_idx].stem
                if frame_id == frame_id_from_img_path:
                    if (set_idx % 10 == 0) and (self.N_vals < self.val_cap):
                        self.val_images.append(str(self.images[set_idx]))
                        self.val_masks.append(str(self.masks[set_idx]))
                        self.val_ids.append(frame_id)
                        self.N_vals += 1
                    else:
                        self.train_images.append(str(self.images[set_idx]))
                        self.train_masks.append(str(self.masks[set_idx]))
                        self.train_ids.append(frame_id)
                        self.N_trains += 1
                else:
                    raise ValueError(f"Mismatch data detected in {self.dataset_name}!")

        print(f"Dataset {self.dataset_name} loaded with {self.N_trains} trains and {self.N_vals} vals.")

    def getItemCount(self):
        return self.N_trains, self.N_vals

    def calcSegMask(self, egoleft, egoright):
        contour_points = []
        for point in egoleft:
            contour_points.append([point[0] * 640, point[1] * 320])
        for i in range(len(egoright)):
            point = egoright[len(egoright) - i - 1]
            contour_points.append([point[0] * 640, point[1] * 320])

        contour = np.array(contour_points, dtype=np.int32)
        binary_segmentation = np.zeros([320, 640], np.float32)
        cv2.drawContours(binary_segmentation, [contour], 0, (255), -1)
        return binary_segmentation

    def calcData(self, ego_left, ego_right, ego_path):
        x_left_lane_offset = ego_left[0][0]
        x_right_lane_offset = ego_right[0][0]
        x_ego_path_offset = ego_path[0][0]
        start_angle = math.atan(
            (ego_path[1][0] * 640 - ego_path[0][0] * 640)
            / abs(ego_path[1][1] * 320 - ego_path[0][1] * 320)
        )
        return [
            x_left_lane_offset,
            x_right_lane_offset,
            x_ego_path_offset,
            start_angle,
        ]

    def getItem(self, index, is_train: bool):
        if is_train:
            frame_id = self.train_ids[index]
            mask_path = self.train_masks[index]
        else:
            frame_id = self.val_ids[index]
            mask_path = self.val_masks[index]

        binary_seg = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        binary_seg = cv2.cvtColor(binary_seg, cv2.COLOR_BGR2RGB)
        binary_seg = binary_seg / 255.0

        return [frame_id, binary_seg]

