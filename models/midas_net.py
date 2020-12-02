"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn

from models.base_model import BaseModel
from models import layers, blocks


class MidasNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = blocks._make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        self.scratch.refinenet4 = layers.FeatureFusionBlock(features)
        self.scratch.refinenet3 = layers.FeatureFusionBlock(features)
        self.scratch.refinenet2 = layers.FeatureFusionBlock(features)
        self.scratch.refinenet1 = layers.FeatureFusionBlock(features)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            layers.Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
        )

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        print("layer_1: ", layer_1.shape)
        print("layer_2: ", layer_2.shape)
        print("layer_3: ", layer_3.shape)
        print("layer_4: ", layer_4.shape)
        print("layer_1_rn: ", layer_1_rn.shape)
        print("layer_2_rn: ", layer_2_rn.shape)
        print("layer_3_rn: ", layer_3_rn.shape)
        print("layer_4_rn: ", layer_4_rn.shape)
        print("path_4: with layer_4_rn ", path_4.shape)
        print("path_3: with path_4, layer_3_rn", path_3.shape)
        print("path_2: with path_3, layer_2_rn", path_2.shape)
        print("path_1: with path_2, layer_1_rn ", path_1.shape)

        out = self.scratch.output_conv(path_1)
        print("out: ", out.shape)

        final_reshape = torch.squeeze(out, dim=1)
        print("Final reshape: ", final_reshape.shape)
        # return torch.squeeze(out, dim=1)
        return final_reshape