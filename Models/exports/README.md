# Export and benchmark EgoLanes model

In order to export the EgoLanes PyTorch checkpoint to FP32 ONNX format, use the `convert_pytorch_to_onnx.py` script:

```bash
python3 Models/exports/convert_pytorch_to_onnx.py \
    -p <PATH TO THE PYTORCH FILE> \
    -o <PATH TO THE OUTPUT ONNX FILE>
```

To benchmark PyTorch and ONNX models on processed validation datasets:

```bash
python3 Models/exports/benchmark_onnx_models.py \
    --pytorch_model_path /path/to/egolanes.pth \
    --fp32_model_path /path/to/egolanes.onnx \
    --dataset_root /path/to/processed_datasets_root \
    --datasets CURVELANES TUSIMPLE
```

