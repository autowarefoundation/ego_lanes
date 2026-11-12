#! /usr/bin/env python3

import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.augmentations import Augmentations
from Models.data_utils.load_data_ego_lanes import VALID_DATASET_LIST
from Models.model_components.ego_lanes.ego_lanes_network import EgoLanesNetwork


class EgoLanesTrainer:
    def __init__(self, checkpoint_path: str = ""):
        # Initializing Data
        self.ego_lanes_seg = None
        self.perspective_image = None
        self.perspective_H = None
        self.perspective_W = None

        # Initializing Tensors
        self.perspective_image_tensor = None
        self.egolanes_tensor = None
        self.pred_binary_seg_tensor = None
        self.total_loss = None

        self.VALID_DATASET_LIST = VALID_DATASET_LIST

        # Checking devices (GPU vs CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for training.")

        # Instantiate model
        self.model = EgoLanesNetwork(pretrained=not bool(checkpoint_path))

        if checkpoint_path:
            print(f"Loading trained EgoLanes model from checkpoint: {checkpoint_path}")
            self.model.load_state_dict(
                torch.load(checkpoint_path, weights_only=True, map_location=self.device)
            )
        else:
            print("Loading vanilla EgoLanes model for training")

        self.model = self.model.to(self.device)

        # TensorBoard
        self.writer = SummaryWriter()

        # Learning rate and optimizer
        self.learning_rate = 0.0001
        self.optimizer = optim.AdamW(self.model.parameters(), self.learning_rate)

        # Loaders
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.seg_loader = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Sobel-like gradient filters for edge preservation loss
        self.gx_filter = (
            torch.tensor(
                [
                    [0.125, 0.0, -0.125],
                    [0.25, 0.0, -0.25],
                    [0.125, 0.0, -0.125],
                ],
                dtype=torch.float32,
            )
            .view((1, 1, 3, 3))
            .to(self.device)
        )

        self.gy_filter = (
            torch.tensor(
                [
                    [0.125, 0.25, 0.125],
                    [0.0, 0.0, 0.0],
                    [-0.125, -0.25, -0.125],
                ],
                dtype=torch.float32,
            )
            .view((1, 1, 3, 3))
            .to(self.device)
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def set_data(self, perspective_image, ego_lanes_seg):
        self.ego_lanes_seg = np.array(ego_lanes_seg)
        self.perspective_image = np.array(perspective_image)
        self.perspective_H, self.perspective_W, _ = self.perspective_image.shape

    def apply_augmentations(self, is_train: bool):
        aug = Augmentations(is_train=is_train, data_type="SEGMENTATION")
        aug.setData(self.perspective_image, self.ego_lanes_seg)
        self.perspective_image, self.ego_lanes_seg = aug.applyTransformSeg(
            self.perspective_image,
            self.ego_lanes_seg,
        )

    def load_data(self):
        # Perspective Image
        perspective_image_tensor = self.image_loader(self.perspective_image)
        perspective_image_tensor = perspective_image_tensor.unsqueeze(0)
        self.perspective_image_tensor = perspective_image_tensor.to(self.device)

        # EgoLanes Segmentation GT
        egolanes_tensor = self.seg_loader(self.ego_lanes_seg).float()
        egolanes_tensor = egolanes_tensor.unsqueeze(0)
        self.egolanes_tensor = egolanes_tensor.to(self.device)

        # Reducing size of EgoLanes Segmentation to 1/4 resolution
        reduction = nn.MaxPool2d(2, stride=2)
        self.egolanes_tensor = reduction(self.egolanes_tensor)
        self.egolanes_tensor = reduction(self.egolanes_tensor)

    def run_model(self):
        self.pred_binary_seg_tensor = self.model(self.perspective_image_tensor)
        self.total_loss = self.calc_ego_lanes_loss()

    def calc_ego_lanes_loss(self):
        pred_ego_left_lane = self.pred_binary_seg_tensor[:, 0, :, :]
        gt_ego_left_lane = self.egolanes_tensor[:, 0, :, :]
        left_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_ego_left_lane, gt_ego_left_lane)
            + self.calc_segmentation_loss(pred_ego_left_lane, gt_ego_left_lane)
        )

        pred_ego_right_lane = self.pred_binary_seg_tensor[:, 1, :, :]
        gt_ego_right_lane = self.egolanes_tensor[:, 1, :, :]
        right_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_ego_right_lane, gt_ego_right_lane)
            + self.calc_segmentation_loss(pred_ego_right_lane, gt_ego_right_lane)
        )

        pred_other_lane = self.pred_binary_seg_tensor[:, 2, :, :]
        gt_other_lane = self.egolanes_tensor[:, 2, :, :]
        other_lane_loss = (
            self.calc_multi_scale_edge_loss(pred_other_lane, gt_other_lane)
            + self.calc_segmentation_loss(pred_other_lane, gt_other_lane)
        )

        ego_lanes_loss = 2 * left_lane_loss + 2 * right_lane_loss + other_lane_loss
        return ego_lanes_loss

    def calc_segmentation_loss(self, pred, gt):
        bce_loss = nn.BCEWithLogitsLoss()
        return bce_loss(pred, gt)

    def calc_multi_scale_edge_loss(self, pred, gt):
        downsample = nn.AvgPool2d(2, stride=2)
        threshold = torch.nn.Threshold(0.0, 1.0, inplace=False)
        relu = nn.ReLU()
        pred = relu(pred)

        prediction_thresholded = threshold(pred)
        prediction_d1 = downsample(prediction_thresholded)
        prediction_d2 = downsample(prediction_d1)
        prediction_d3 = downsample(prediction_d2)
        prediction_d4 = downsample(prediction_d3)

        gt_d1 = downsample(gt)
        gt_d2 = downsample(gt_d1)
        gt_d3 = downsample(gt_d2)
        gt_d4 = downsample(gt_d3)

        edge_loss_d0 = self.calc_edge_loss(pred, gt)
        edge_loss_d1 = self.calc_edge_loss(prediction_d1, gt_d1)
        edge_loss_d2 = self.calc_edge_loss(prediction_d2, gt_d2)
        edge_loss_d3 = self.calc_edge_loss(prediction_d3, gt_d3)
        edge_loss_d4 = self.calc_edge_loss(prediction_d4, gt_d4)

        return (edge_loss_d0 + edge_loss_d1 + edge_loss_d2 + edge_loss_d3 + edge_loss_d4) / 5

    def calc_edge_loss(self, prediction, gt):
        g_x_pred = nn.functional.conv2d(prediction, self.gx_filter, padding=1)
        g_y_pred = nn.functional.conv2d(prediction, self.gy_filter, padding=1)

        g_x_gt = nn.functional.conv2d(gt, self.gx_filter, padding=1)
        g_y_gt = nn.functional.conv2d(gt, self.gy_filter, padding=1)

        edge_diff_mae = torch.abs(g_x_pred - g_x_gt) + torch.abs(g_y_pred - g_y_gt)
        return torch.mean(edge_diff_mae)

    def loss_backward(self):
        self.total_loss.backward()

    def get_total_loss(self):
        return self.total_loss.item()

    def get_validation_loss_value(self):
        return self.total_loss.detach().cpu().numpy()

    def log_loss(self, log_count):
        self.writer.add_scalar("Loss", self.get_total_loss(), log_count)

    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_train_mode(self):
        self.model = self.model.train()

    def set_eval_mode(self):
        self.model = self.model.eval()

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print("Finished training")

    def save_visualization(self, log_count, vis_path: str = "", is_train: bool = False):
        fig_lane_seg, axs_seg = plt.subplots(2, 1, figsize=(8, 8))
        fig_raw_lane_seg, axs_raw_seg = plt.subplots(2, 1, figsize=(8, 8))

        alpha = 0.5

        vis_predict_object = cv2.resize(np.array(self.perspective_image), (160, 80))
        gt_object = cv2.resize(np.array(self.perspective_image), (160, 80))

        vis_raw_predict_object = np.zeros((80, 160, 3), dtype="uint8")
        gt_raw_object = np.zeros((80, 160, 3), dtype="uint8")

        egolanes_prediction = torch.squeeze(self.pred_binary_seg_tensor, 0).cpu().detach().numpy()
        egolanes_gt = torch.squeeze(self.egolanes_tensor, 0).cpu().detach().numpy()

        pred_egoleft_lanes = np.where(egolanes_prediction[0, :, :] > 0)
        pred_egoright_lanes = np.where(egolanes_prediction[1, :, :] > 0)
        pred_other_lanes = np.where(egolanes_prediction[2, :, :] > 0)

        gt_egoleft_lanes = np.where(egolanes_gt[0, :, :] > 0)
        gt_egoright_lanes = np.where(egolanes_gt[1, :, :] > 0)
        gt_other_lanes = np.where(egolanes_gt[2, :, :] > 0)

        # EgoLeft Lane
        vis_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], :] = [0, 255, 255]
        gt_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], :] = [0, 255, 255]
        vis_raw_predict_object[pred_egoleft_lanes[0], pred_egoleft_lanes[1], :] = [0, 255, 255]
        gt_raw_object[gt_egoleft_lanes[0], gt_egoleft_lanes[1], :] = [0, 255, 255]

        # EgoRight Lane
        vis_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], :] = [255, 0, 200]
        gt_object[gt_egoright_lanes[0], gt_egoright_lanes[1], :] = [255, 0, 200]
        vis_raw_predict_object[pred_egoright_lanes[0], pred_egoright_lanes[1], :] = [255, 0, 200]
        gt_raw_object[gt_egoright_lanes[0], gt_egoright_lanes[1], :] = [255, 0, 200]

        # Other Lanes
        vis_predict_object[pred_other_lanes[0], pred_other_lanes[1], :] = [0, 255, 145]
        gt_object[gt_other_lanes[0], gt_other_lanes[1], :] = [0, 255, 145]
        vis_raw_predict_object[pred_other_lanes[0], pred_other_lanes[1], :] = [0, 255, 145]
        gt_raw_object[gt_other_lanes[0], gt_other_lanes[1], :] = [0, 255, 145]

        vis_predict_object = cv2.resize(vis_predict_object, (640, 320), interpolation=cv2.INTER_NEAREST)
        gt_object = cv2.resize(gt_object, (640, 320), interpolation=cv2.INTER_NEAREST)

        prediction_vis = cv2.addWeighted(vis_predict_object, alpha, self.perspective_image, 1 - alpha, 0)
        gt_vis = cv2.addWeighted(gt_object, alpha, self.perspective_image, 1 - alpha, 0)

        axs_seg[0].set_title("Prediction", fontweight="bold")
        axs_seg[0].imshow(prediction_vis)
        axs_seg[1].set_title("Ground Truth", fontweight="bold")
        axs_seg[1].imshow(gt_vis)
        if is_train:
            self.writer.add_figure("Train (Seg)", fig_lane_seg, global_step=log_count)
        else:
            fig_lane_seg.savefig(vis_path + "_seg.png")
        plt.close(fig_lane_seg)

        axs_raw_seg[0].set_title("Prediction (RAW)", fontweight="bold")
        axs_raw_seg[0].imshow(vis_raw_predict_object)
        axs_raw_seg[1].set_title("Ground Truth (RAW)", fontweight="bold")
        axs_raw_seg[1].imshow(gt_raw_object)
        if is_train:
            self.writer.add_figure("Train (Seg) RAW", fig_raw_lane_seg, global_step=log_count)
        else:
            fig_raw_lane_seg.savefig(vis_path + "_seg_raw.png")
        plt.close(fig_raw_lane_seg)

    def log_validation_dataset(self, dataset, validation_loss_dataset_total, log_count):
        self.writer.add_scalar(f"{dataset} (Validation)", validation_loss_dataset_total, log_count)

    def log_validation_overall(self, overall_val_score, log_count):
        self.writer.add_scalar("Overall (Validation)", overall_val_score, log_count)

