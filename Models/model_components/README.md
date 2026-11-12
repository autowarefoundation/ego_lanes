# model_components

Contains PyTorch classes for all components of the `EgoLanesNetwork` semantic lane boundary segmentation model:

- `ego_lanes/backbone.py` — EfficientNet-B0 encoder extracting 5 multi-scale feature maps (`[l0, l2, l3, l4, l8]`).
- `ego_lanes/backbone_feature_fusion.py` — Multi-scale pooling and channel concatenation into a 1456-channel deep feature block.
- `ego_lanes/auto_steer_context.py` — Global pooling, MLP context bottleneck, spatial reshaping, and residual attention (`context * features + features`).
- `ego_lanes/ego_path_neck.py` — Decoder neck with transposed-convolution upsampling and encoder skip connections.
- `ego_lanes/ego_lanes_head.py` — 3-class segmentation head outputting logits at 1/4 input resolution (`ego-left`, `ego-right`, `other lanes`).
- `ego_lanes/ego_lanes_network.py` — Full end-to-end `EgoLanesNetwork` module.

