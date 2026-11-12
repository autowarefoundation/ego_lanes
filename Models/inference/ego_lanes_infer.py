import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Models.model_components.ego_lanes.ego_lanes_network import EgoLanesNetwork


class EgoLanesNetworkInfer:
    def __init__(self, checkpoint_path: str = ""):
        self.train_size = (640, 320)  # (width, height)

        # Image loader (ImageNet RGB normalization)
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} for inference")

        # Initialize model (skip ImageNet backbone download when loading a full checkpoint)
        self.model = EgoLanesNetwork(pretrained=not bool(checkpoint_path))
        if checkpoint_path:
            print(f"Loading trained EgoLanes checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(
                    checkpoint_path,
                    weights_only=True,
                    map_location=self.device,
                )
            except Exception:
                checkpoint = torch.load(
                    checkpoint_path,
                    weights_only=False,
                    map_location=self.device,
                )

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = (
                    checkpoint["model"].state_dict()
                    if hasattr(checkpoint["model"], "state_dict")
                    else checkpoint["model"]
                )
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
        else:
            print("Loading vanilla EgoLanes model (no checkpoint provided)")

        self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if pil_img.size != self.train_size:
            pil_img = pil_img.resize(self.train_size, Image.BILINEAR)
        return pil_img

    @torch.no_grad()
    def inference(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Run EgoLanes inference on an RGB image (PIL.Image or HWC uint8 numpy array).

        Returns:
            np.ndarray of shape (3, 80, 160) with raw channel logits:
            [0] ego-left lane, [1] ego-right lane, [2] other lanes.
        """
        pil_img = self._prepare_image(image)
        image_tensor = self.image_loader(pil_img).unsqueeze(0).to(self.device)
        prediction = self.model(image_tensor)
        return prediction.squeeze(0).cpu().numpy()


class EgoLanesONNXInfer:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.train_size = (640, 320)  # (width, height)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"ONNX model loaded with provider: {self.session.get_providers()[0]}")

    def _prepare_array(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        if pil_img.size != self.train_size:
            pil_img = pil_img.resize(self.train_size, Image.BILINEAR)

        img_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        img_norm = (img_np - self.mean) / self.std
        return img_norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def inference(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        input_tensor = self._prepare_array(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0].squeeze(0)

