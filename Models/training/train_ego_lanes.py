#! /usr/bin/env python3

import argparse
import os
import pathlib
import random
import sys

import torch
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.data_utils.load_data_ego_lanes import LoadDataEgoLanes, VALID_DATASET_LIST
from Models.training.ego_lanes_trainer import EgoLanesTrainer

PERSPECTIVE_JSON_PATH = "drivable_path.json"
PERSPECTIVE_IMG_PATH = "image"
PERSPECTIVE_VIS_PATH = "visualization"
PERSPECTIVE_MASK_PATH = "mask"


def parse_args():
    parser = argparse.ArgumentParser(description="Train EgoLanes semantic lane segmentation network")
    parser.add_argument(
        "-r",
        "--dataset_root",
        type=str,
        required=True,
        help="Root directory containing processed datasets (e.g., TUSIMPLE/, CURVELANES/, OPENLANE/, ...)",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="runs/ego_lanes",
        help="Directory where model checkpoints and validation figures will be saved (default: runs/ego_lanes)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional pretrained checkpoint (.pth) path to resume or fine-tune from",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CURVELANES", "TUSIMPLE"],
        choices=VALID_DATASET_LIST,
        help=f"Processed dataset subdirectories under dataset_root to train on (default: CURVELANES TUSIMPLE)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=8, help="Gradient accumulation batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--log_step_loss", type=int, default=2500, help="TensorBoard loss logging interval")
    parser.add_argument("--log_step_vis", type=int, default=10000, help="TensorBoard visualization logging interval")
    parser.add_argument("--log_step_model", type=int, default=15000, help="Checkpoint & validation interval")
    parser.add_argument("--num_val_vis", type=int, default=25, help="Validation visualization samples per validation pass")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = args.dataset_root
    active_datasets = args.datasets

    model_save_root_path = os.path.join(args.save_dir, "models")
    os.makedirs(model_save_root_path, exist_ok=True)

    vis_save_root_path = os.path.join(args.save_dir, "figures")
    os.makedirs(vis_save_root_path, exist_ok=True)

    # Init metadata for datasets
    msdict = {}
    for dataset in active_datasets:
        msdict[dataset] = {
            "path_labels": os.path.join(dataset_root, dataset, PERSPECTIVE_JSON_PATH),
            "path_masks": os.path.join(dataset_root, dataset, PERSPECTIVE_MASK_PATH),
            "path_perspective_vis": os.path.join(dataset_root, dataset, PERSPECTIVE_VIS_PATH),
            "path_perspective_image": os.path.join(dataset_root, dataset, PERSPECTIVE_IMG_PATH),
        }

    # Load datasets
    for dataset in active_datasets:
        this_dataset_loader = LoadDataEgoLanes(
            labels_filepath=msdict[dataset]["path_labels"],
            mask_dirpath=msdict[dataset]["path_masks"],
            images_filepath=msdict[dataset]["path_perspective_image"],
            dataset=dataset,
        )
        n_trains, n_vals = this_dataset_loader.getItemCount()
        random_sample_list = random.sample(list(range(0, n_trains)), n_trains)

        msdict[dataset]["loader"] = this_dataset_loader
        msdict[dataset]["N_trains"] = n_trains
        msdict[dataset]["N_vals"] = n_vals
        msdict[dataset]["sample_list"] = random_sample_list

        print(f"LOADED: {dataset} with {n_trains} train samples, {n_vals} val samples.")

    msdict["Nsum_trains"] = sum(msdict[d]["N_trains"] for d in active_datasets)
    print(f"Total train samples: {msdict['Nsum_trains']}")

    msdict["Nsum_vals"] = sum(msdict[d]["N_vals"] for d in active_datasets)
    print(f"Total val samples: {msdict['Nsum_vals']}")

    # Trainer instance
    trainer = EgoLanesTrainer(checkpoint_path=args.checkpoint_path) if args.checkpoint_path else EgoLanesTrainer()
    trainer.zero_grad()

    print("Beginning Training")

    for epoch in range(args.epochs):
        data_list = active_datasets.copy()
        print(f"EPOCH : {epoch}")

        trainer.set_learning_rate(args.lr)
        apply_augmentation = True

        random.shuffle(data_list)
        msdict["data_list_count"] = 0
        msdict["sample_counter"] = 0

        for dataset in active_datasets:
            msdict[dataset]["iter"] = 0
            msdict[dataset]["completed"] = False

        while True:
            msdict["sample_counter"] += 1
            msdict["log_counter"] = msdict["sample_counter"] + msdict["Nsum_trains"] * epoch

            for dataset in active_datasets:
                n_trains = msdict[dataset]["N_trains"]
                if msdict[dataset]["iter"] >= n_trains - 1:
                    if not msdict[dataset]["completed"] and dataset in data_list:
                        data_list.remove(dataset)
                    msdict[dataset]["completed"] = True

            if all(msdict[d]["completed"] for d in active_datasets):
                break

            if msdict["data_list_count"] >= len(data_list):
                msdict["data_list_count"] = 0

            current_dataset = data_list[msdict["data_list_count"]]
            current_dataset_iter = msdict[current_dataset]["iter"]
            frame_id, ego_lanes_seg = msdict[current_dataset]["loader"].getItem(
                msdict[current_dataset]["sample_list"][current_dataset_iter],
                is_train=True,
            )
            msdict[current_dataset]["iter"] = current_dataset_iter + 1

            perspective_image = Image.open(
                os.path.join(msdict[current_dataset]["path_perspective_image"], f"{frame_id}.jpg")
            ).convert("RGB")

            trainer.set_data(perspective_image, ego_lanes_seg)
            trainer.apply_augmentations(apply_augmentation)
            trainer.load_data()
            trainer.run_model()
            trainer.loss_backward()

            if (msdict["sample_counter"] + 1) % args.batch_size == 0:
                trainer.run_optimizer()

            if (msdict["sample_counter"] + 1) % args.log_step_loss == 0:
                trainer.log_loss(msdict["log_counter"] + 1)

            if (msdict["sample_counter"] + 1) % args.log_step_vis == 0:
                trainer.save_visualization(msdict["log_counter"] + 1, is_train=True)

            if (msdict["sample_counter"] + 1) % args.log_step_model == 0:
                print(f"\nIteration: {msdict['log_counter'] + 1}")
                print("================ Saving Model ================")

                model_save_path = os.path.join(
                    model_save_root_path,
                    f"iter_{msdict['log_counter'] + 1}_epoch_{epoch}_step_{msdict['sample_counter'] + 1}.pth",
                )
                trainer.save_model(model_save_path)
                trainer.set_eval_mode()

                for dataset in active_datasets:
                    msdict[dataset]["num_val_samples"] = 0
                    msdict[dataset]["total_running"] = 0

                with torch.no_grad():
                    print("================ Running validation calculation ================")

                    for dataset in active_datasets:
                        for val_count in range(msdict[dataset]["N_vals"]):
                            frame_id, ego_lanes_seg = msdict[dataset]["loader"].getItem(
                                val_count,
                                is_train=False,
                            )
                            msdict[dataset]["num_val_samples"] += 1

                            perspective_image = Image.open(
                                os.path.join(msdict[dataset]["path_perspective_image"], f"{frame_id}.jpg")
                            ).convert("RGB")

                            trainer.set_data(perspective_image, ego_lanes_seg)
                            trainer.apply_augmentations(False)
                            trainer.load_data()
                            trainer.run_model()

                            msdict[dataset]["total_running"] += trainer.get_validation_loss_value()

                            if val_count < args.num_val_vis:
                                vis_path = os.path.join(
                                    vis_save_root_path,
                                    f"epoch_{epoch}_step_{msdict['log_counter'] + 1}_image_{dataset}_{frame_id}",
                                )
                                trainer.save_visualization(
                                    msdict["log_counter"] + 1 + val_count,
                                    vis_path,
                                    is_train=False,
                                )

                    overall_val_score = 0
                    for dataset in active_datasets:
                        validation_loss_dataset_total = (
                            msdict[dataset]["total_running"] / msdict[dataset]["num_val_samples"]
                        )
                        overall_val_score += validation_loss_dataset_total
                        print("DATASET :", dataset, " VAL SCORE : ", validation_loss_dataset_total)
                        trainer.log_validation_dataset(
                            dataset,
                            validation_loss_dataset_total,
                            msdict["log_counter"] + 1,
                        )

                    overall_val_score = overall_val_score / len(active_datasets)
                    print("OVERALL VAL SCORE :", overall_val_score)
                    trainer.log_validation_overall(overall_val_score, msdict["log_counter"] + 1)

                print("================ Continuing with training ================")
                trainer.set_train_mode()

            msdict["data_list_count"] += 1

    trainer.cleanup()


if __name__ == "__main__":
    main()

