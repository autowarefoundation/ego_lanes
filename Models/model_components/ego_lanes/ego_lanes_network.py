import torch.nn as nn

from .backbone import Backbone
from .backbone_feature_fusion import BackboneFeatureFusion
from .auto_steer_context import AutoSteerContext
from .ego_path_neck import EgoPathNeck
from .ego_lanes_head import EgoLanesHead


class EgoLanesNetwork(nn.Module):
    def __init__(self, pretrained: bool = True):
        super(EgoLanesNetwork, self).__init__()

        # Upstream blocks
        self.BEVBackbone = Backbone(pretrained=pretrained)

        # Feature Fusion
        self.BackboneFeatureFusion = BackboneFeatureFusion()

        # BEV Path Context
        self.AutoSteerContext = AutoSteerContext()

        # EgoPath Neck
        self.EgopathNeck = EgoPathNeck()

        # EgoLanes Head
        self.EgoLanesHead = EgoLanesHead()

    def forward(self, image):
        features = self.BEVBackbone(image)
        fused_features = self.BackboneFeatureFusion(features)
        context = self.AutoSteerContext(fused_features)
        neck = self.EgopathNeck(context, features)
        ego_lanes = self.EgoLanesHead(neck)

        return ego_lanes

