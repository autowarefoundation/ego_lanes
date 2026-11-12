# EgoLanes Training

## train_ego_lanes.py

Main script for training the EgoLanes neural network on processed multi-dataset lane segmentation data.

### Example usage

```bash
python3 Models/training/train_ego_lanes.py \
    --dataset_root /path/to/processed_datasets_root \
    --datasets CURVELANES TUSIMPLE \
    --save_dir runs/ego_lanes
```

### Main parameters:

- `-r`, `--dataset_root` : processed dataset root directory (required)
- `-s`, `--save_dir` : output directory for model checkpoints and validation figures (default: `runs/ego_lanes`)
- `-c`, `--checkpoint_path` : optional pretrained `.pth` checkpoint path to resume/fine-tune from
- `--datasets` : list of processed dataset folders under `dataset_root` (`CURVELANES`, `TUSIMPLE`, `OPENLANE`, `JIQING`, `ONCE3DLANE`, `CULANE`, `BDD100K`, `COMMA2K19`, `ROADWORK`)
- `--epochs` : number of training epochs (default: `10`)
- `--batch_size` : gradient accumulation batch size (default: `8`)
- `--lr` : learning rate (default: `1e-4`)
- `--log_step_loss` : TensorBoard loss logging interval in steps (default: `2500`)
- `--log_step_vis` : TensorBoard image logging interval in steps (default: `10000`)
- `--log_step_model` : checkpoint save & validation interval in steps (default: `15000`)
- `--num_val_vis` : number of validation visualization images saved per pass (default: `25`)

## ego_lanes_trainer.py

Helper class (`EgoLanesTrainer`) implementing model initialization, data loading, `SEGMENTATION` augmentations, weighted BCE + multi-scale edge preservation loss, TensorBoard logging, and visualization export.

