## ego_lanes_infer.py

Helper classes for inference using the EgoLanes neural network:

- `EgoLanesNetworkInfer`: PyTorch `.pth` inference wrapper (`EgoLanesNetwork`).
- `EgoLanesONNXInfer`: ONNX Runtime `.onnx` inference wrapper.

### Example usage

```python
from PIL import Image
from Models.inference.ego_lanes_infer import EgoLanesNetworkInfer

model = EgoLanesNetworkInfer("/path/to/egolanes.pth")
img = Image.open("Media/highway_normal_1.png").convert("RGB")
seg_logits = model.inference(img)  # shape: (3, 80, 160)
```

