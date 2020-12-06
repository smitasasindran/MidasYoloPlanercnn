"""MidashNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import torch
import torch.nn as nn
import re

from models.base_model import BaseModel
from models import blocks, yolo3_net, planercnn_net
from models import layers
from utils.torch_utils import model_info
from utils.planercnn_config import Config

class MidasYoloNet(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, features=256, non_negative=True, yolo_cfg='',
                 augment=False, image_size=None, device='cpu'):
        """Init.
        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to resnet50
        """
        print("Loading weights: ", path)

        super(MidasYoloNet, self).__init__()

        use_pretrained = False if path is None else True

        self.pretrained, self.scratch = blocks._make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=use_pretrained)

        # Midas Decoder part
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

        self.yolo_head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),  # 208 x 208
            nn.BatchNorm2d(32, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False), # 104 x 104
            nn.BatchNorm2d(64, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 52 x 52
            nn.BatchNorm2d(128, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 26 x 26
            nn.BatchNorm2d(256, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 13 x 13
            nn.BatchNorm2d(512, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),  # 13 x 13
            nn.BatchNorm2d(1024, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1, bias=False),  # 13 x 13
            nn.BatchNorm2d(2048, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)

        )
        # Concat midas output with yolo head output
        self.yolo_connect = nn.Sequential(
            nn.Conv2d(2048+256, 2048, kernel_size=3, stride=1, padding=1, bias=False),  # 13 x 13
            nn.BatchNorm2d(2048, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.yolo_decoder = yolo3_net.Darknet(cfg=yolo_cfg, img_size=image_size)
        # self.yolo_decoder = yolo3_net.YoloDecoder(img_size=image_size, features=features)

        self.yolo_layers = self.yolo_decoder.yolo_layers
        # self.module_list = self.yolo_decoder.module_list # ToDo Smita: fix this, its a hack

        # Add planercnn model. Can directly add, as its backbone is also Resnet101, same as midas
        planercnn_config = Config(None)
        self.planercnn_decoder = planercnn_net.MaskRCNN(planercnn_config, device)


    def modulelist(self):
        return self.yolo_decoder.module_list

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        # print("Input size: ", x.shape)
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # 52 x 52
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # 26 x 26
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # 13 x 13

        # Midas Decoder
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Midas output
        out = self.scratch.output_conv(path_1)
        midas_output = torch.squeeze(out, dim=1)

        # Feed to yolo decoder
        # Layer 83 route to e3 => 82/80/78/77, 13x13  -- layer_4_rn
        # Layer 93 route to e2=> 61, 26x26            -- layer_3_rn
        # Layer 105 route to e1=> 36, 52x52 =>        -- layer_2_rn

        # print("Before yolo decoder call")
        yolo_head = self.yolo_head(x)
        combined = torch.cat((yolo_head, layer_4_rn), 1)
        yolo_connect = self.yolo_connect(combined)
        yolo_decoder = self.yolo_decoder(yolo_connect, layer_2_rn, layer_3_rn)

        # Planercnn decoder takes raw input, and then the three encoder inputs. Called using predict function
        # mode = 'training' if self.training else 'inference'
        # planercnn_out = self.planercnn_decoder.predict(x, mode, encoder_inps=[layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn])

        # ToDo: Send out Planercnn output too
        return midas_output, yolo_decoder


    def freeze_layers(self, keys):
        """Sets model layers as frozen if their names match the given regular expression."""
        layer_regexes = {
            "encoder": r"(pretrained.*)|(scratch.layer.*)",
            "midas": r"(scratch.refinenet.*)|(scratch.output.*)",
            "yolo": r"(yolo_head.*)|(yolo_connect.*)|(yolo_decoder.*)",
            "planercnn": r"(planercnn_head.*)|(planercnn_decoder.*)"
        }

        # First set everything to true, then freeze layers
        for param in self.named_parameters():
            param[1].requires_grad = True

        # If layer needs to be frozen, set requires_grad to false
        for key in keys:
            layer_regex = layer_regexes[key]
            print("Layer regex: ", str(layer_regex))

            for param in self.named_parameters():
                layer_name = param[0]
                freeze = bool(re.fullmatch(layer_regex, layer_name))
                if freeze:
                    param[1].requires_grad = False


    def load_model_weights(self, path):
        pass


    def load_midas_weights(self, path):
        # Encoder + decoder
        self.load(path)

    def load_planercnn_weights(self, path):
        self.planercnn_decoder.load_weights(path)


    def load_yolo_weights(self, path, device):
        mappings = self.yolo_decoder.yolo3_layer_weight_mappings()
        yolo_state_dict = torch.load(path, map_location=device)['model']
        yolo_weight_dict = {k: v for k, v in yolo_state_dict.items() if k in mappings.values()}

        yolo_dict = {}
        try:
            model_dict = self.state_dict()
            print("Yolo weight dict ultralytics filtered=", yolo_weight_dict.keys())
            for k, v in mappings.items():
                if (model_dict[k].numel() == yolo_weight_dict[v].numel()):
                    yolo_dict[k] = yolo_weight_dict[v]

            model_dict.update(yolo_dict)
            self.load_state_dict(model_dict, strict=False)
        except KeyError as e:
            s = "Yolo weights {0} are not compatible.".format(path)
            raise KeyError(s) from e


    def info(self, verbose=False):
        model_info(self, verbose)
