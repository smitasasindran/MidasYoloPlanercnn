import torch
import torch.nn as nn
import numpy as np
from utils.yolo_utils import parse_model_cfg, get_yolo_layers
from utils.torch_utils import fuse_conv_and_bn, model_info, scale_img
from models.blocks import create_modules
from models.layers import YOLOLayer, FeatureConcat

ONNX_EXPORT = False

# ToDo: Delete this class, using modified DarkNet below
class YoloDecoder(nn.Module):
    """
    Yolo3 decoder model - Either use this or Darknet below
    The decoder layers start from layer #83
    There are three connections from Yolo encoder to the decoder part, namely from layers 36, 61 and 82
    Before the decoder layer, there is a SPP (Spacial Pyramid Pooling) block, which is not included
    """

    def __init__(self, img_size=(416, 416), features=256):
        super(YoloDecoder, self).__init__()
        nc = 4 # ToDo: Number of classes, Don't hardcode
        # Anchors, masks, stride is hardcoded in get_yolo function, pick from config
        # Yolo3 layers have routes, which are concatenation of some layers -- from config file
        yolo_index = -1
        self.module_list = nn.ModuleList()

        # Picking yolo3 layers after the maxpooling section
        self.y_84 = self.single_cbl(2048, 512, 1)
        self.y_85 = self.single_cbl(512, 1024, 3)
        self.y_86 = self.single_cbl(1024, 512, 1)
        self.y_87 = self.single_cbl(512, 1024, 3)
        self.y_88 = self.pre_yolo(1024, 27, 1)   # Pre-yolo layer. Calculate channel size of 27, don't hardcode. No batchnorm..

        # Yolo Layer
        yolo_index += 1
        self.y_89 = self.get_yolo(nc, yolo_index, img_size)

        # self.y_90 = FeatureConcat(layers=[-4]) # == 86 Define empty layer, and just return 86 output
        self.y_91 = self.single_cbl(512, 256, 1)
        self.y_92 = nn.Upsample(scale_factor=2)
        # self.y_93 = FeatureConcat(layers=[-1, 61]) # Encoder, Do this during forward pass

        self.y_94 = self.single_cbl(768, 256, 1)
        self.y_95 = self.single_cbl(256, 512, 3)
        self.y_96 = self.single_cbl(512, 256, 1)
        self.y_97 = self.single_cbl(256, 512, 3)
        self.y_98 = self.single_cbl(512, 256, 1)
        self.y_99 = self.single_cbl(256, 512, 3)
        self.y_100 = self.pre_yolo(512, 27, 1)

        # Yolo layer
        yolo_index += 1
        self.y_101 = self.get_yolo(nc, yolo_index, img_size)

        # self.y_102 = FeatureConcat(-4) # == 98
        self.y_103 = self.single_cbl(256, 128, 1)
        self.y_104 = nn.Upsample(scale_factor=2) #, mode=nearest
        # self.y_105 = FeatureConcat(-1, 36)  # Encoder

        self.y_106 = self.single_cbl(384, 128, 1)
        self.y_107 = self.single_cbl(128, 256, 3)
        self.y_108 = self.single_cbl(256, 128, 1)
        self.y_109 = self.single_cbl(128, 256, 3)
        self.y_110 = self.single_cbl(256, 128, 1)
        self.y_111 = self.single_cbl(128, 256, 3)
        self.y_112 = self.pre_yolo(256, 27, 1)

        # Yolo layer
        yolo_index += 1
        self.y_113 = self.get_yolo(nc, yolo_index, img_size)

        self.yolo_layers = get_yolo_layers(self)

        # ToDo: Populate this with all layers. Verify if this is correct
        self.module_list.extend([self.y_84, self.y_85, self.y_86, self.y_87, self.y_88, self.y_89,
                            self.y_91, self.y_92, self.y_94, self.y_95, self.y_96, self.y_97,
                            self.y_98, self.y_99, self.y_100, self.y_101, self.y_103, self.y_104,
                            self.y_106, self.y_107, self.y_108, self.y_109, self.y_110, self.y_111,
                            self.y_112]) #self.y_90, self.y_93, self.y_102, self.y_105,


    def forward(self, *xs):
        x = xs[0]
        encoder_1 = xs[1] # From layer 36
        encoder_2 = xs[2] # From layer 61

        print("Input size: ", x.shape)
        # First section. Input from yolo head also mixes in raw image input
        x = self.y_84(x)
        x = self.y_85(x)
        x_86 = self.y_86(x)
        x = self.y_87(x_86)
        x = self.y_88(x)

        # First yolo layer + upsample
        yolo_1 = self.y_89(x) # yolo layer, decide how many inputs to pass..
        # x = self.y_90(x_86) # input from 86, route -4
        x = x_86
        x = self.y_91(x)
        x = self.y_92(x)
        # x = self.y_93(x, encoder_2) # input from -1, 61 (ie encoder 2)
        x = torch.cat((x, encoder_2), 1)

        # Second section
        x = self.y_94(x)
        x = self.y_95(x)
        x = self.y_96(x)
        x = self.y_97(x)
        x_98 = self.y_98(x)
        x = self.y_99(x_98)
        x = self.y_100(x)

        # Second yolo + upsample
        yolo_2 = self.y_101(x)
        # x = self.y_102(x_98) # input from 98, route -4
        x = x_98
        x = self.y_103(x)
        x = self.y_104(x)
        # x = self.y_105(x, encoder_1) # input from -1, 36 (ie encoder 1)
        x = torch.cat((x, encoder_1))

        # Third section
        x = self.y_106(x)
        x = self.y_107(x)
        x = self.y_108(x)
        x = self.y_109(x)
        x = self.y_110(x)
        x = self.y_111(x)
        x = self.y_112(x)

        # Final yolo layer
        yolo_3 = self.y_113(x)

        yolo_out = [yolo_1, yolo_2, yolo_3]
        if self.training:  # train
            return yolo_out

        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p


    def single_cbl(self, in_channels, out_channels, kernel_size):
        # ToDo Smita: Some layers do not have batchnorm, fix accordingly
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),  # 13 x 13
            nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.0001),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def pre_yolo(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
        )

    def get_yolo(self, nc, yolo_index, img_size):
        # Anchors and mask from yolo3 config file
        anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]])
        mask = [[6, 7, 8],
                [3, 4, 5],
                [0, 1, 2]] # mask1, mask2, mask3
        stride = [32, 16, 8, 4, 2][yolo_index]  # P3-P7 stride

        # layers = mdef['from'] if 'from' in mdef else []
        layer = YOLOLayer(anchors=anchors[mask[yolo_index]],  # anchor list
                              nc=nc,  # number of classes
                              img_size=img_size,  # (416, 416)
                              yolo_index=yolo_index,  # 0, 1, 2...
                              layers=[],  # output layers
                              stride=stride)
        return layer

    def yolo3_layer_weight_mappings(self):
        mapping = {
            'yolo_decoder.module_list.0.Conv2d.weight':      'module_list.84.Conv2d.weight',
            'yolo_decoder.module_list.0.BatchNorm2d.weight': 'module_list.84.BatchNorm2d.weight',
            'yolo_decoder.module_list.0.BatchNorm2d.bias':   'module_list.84.BatchNorm2d.bias',
            'yolo_decoder.module_list.1.Conv2d.weight':      'module_list.85.Conv2d.weight',
            'yolo_decoder.module_list.1.BatchNorm2d.weight': 'module_list.85.BatchNorm2d.weight',
            'yolo_decoder.module_list.1.BatchNorm2d.bias':   'module_list.85.BatchNorm2d.bias',
            'yolo_decoder.module_list.2.Conv2d.weight':      'module_list.86.Conv2d.weight',
            'yolo_decoder.module_list.2.BatchNorm2d.weight': 'module_list.86.BatchNorm2d.weight',
            'yolo_decoder.module_list.2.BatchNorm2d.bias':   'module_list.86.BatchNorm2d.bias',
            'yolo_decoder.module_list.3.Conv2d.weight':      'module_list.87.Conv2d.weight',
            'yolo_decoder.module_list.3.BatchNorm2d.weight': 'module_list.87.BatchNorm2d.weight',
            'yolo_decoder.module_list.3.BatchNorm2d.bias':   'module_list.87.BatchNorm2d.bias',
            'yolo_decoder.module_list.4.Conv2d.weight':      'module_list.88.Conv2d.weight',
            'yolo_decoder.module_list.4.Conv2d.bias':        'module_list.88.Conv2d.bias',
            'yolo_decoder.module_list.7.Conv2d.weight':      'module_list.91.Conv2d.weight',
            'yolo_decoder.module_list.7.BatchNorm2d.weight': 'module_list.91.BatchNorm2d.weight',
            'yolo_decoder.module_list.7.BatchNorm2d.bias':   'module_list.91.BatchNorm2d.bias',
            'yolo_decoder.module_list.10.Conv2d.weight':     'module_list.94.Conv2d.weight',
            'yolo_decoder.module_list.10.BatchNorm2d.weight':'module_list.94.BatchNorm2d.weight',
            'yolo_decoder.module_list.10.BatchNorm2d.bias':  'module_list.94.BatchNorm2d.bias',
            'yolo_decoder.module_list.11.Conv2d.weight':     'module_list.95.Conv2d.weight',
            'yolo_decoder.module_list.11.BatchNorm2d.weight':'module_list.95.BatchNorm2d.weight',
            'yolo_decoder.module_list.11.BatchNorm2d.bias':  'module_list.95.BatchNorm2d.bias',
            'yolo_decoder.module_list.12.Conv2d.weight':     'module_list.96.Conv2d.weight',
            'yolo_decoder.module_list.12.BatchNorm2d.weight':'module_list.96.BatchNorm2d.weight',
            'yolo_decoder.module_list.12.BatchNorm2d.bias':  'module_list.96.BatchNorm2d.bias',
            'yolo_decoder.module_list.13.Conv2d.weight':     'module_list.97.Conv2d.weight',
            'yolo_decoder.module_list.13.BatchNorm2d.weight':'module_list.97.BatchNorm2d.weight',
            'yolo_decoder.module_list.13.BatchNorm2d.bias':  'module_list.97.BatchNorm2d.bias',
            'yolo_decoder.module_list.14.Conv2d.weight':     'module_list.98.Conv2d.weight',
            'yolo_decoder.module_list.14.BatchNorm2d.weight':'module_list.98.BatchNorm2d.weight',
            'yolo_decoder.module_list.14.BatchNorm2d.bias':  'module_list.98.BatchNorm2d.bias',
            'yolo_decoder.module_list.15.Conv2d.weight':     'module_list.99.Conv2d.weight',
            'yolo_decoder.module_list.15.BatchNorm2d.weight':'module_list.99.BatchNorm2d.weight',
            'yolo_decoder.module_list.15.BatchNorm2d.bias':  'module_list.99.BatchNorm2d.bias',
            'yolo_decoder.module_list.16.Conv2d.weight':     'module_list.100.Conv2d.weight',
            'yolo_decoder.module_list.16.Conv2d.bias':       'module_list.100.Conv2d.bias',
            'yolo_decoder.module_list.19.Conv2d.weight':     'module_list.103.Conv2d.weight',
            'yolo_decoder.module_list.19.BatchNorm2d.weight':'module_list.103.BatchNorm2d.weight',
            'yolo_decoder.module_list.19.BatchNorm2d.bias':  'module_list.103.BatchNorm2d.bias',
            'yolo_decoder.module_list.22.Conv2d.weight':     'module_list.106.Conv2d.weight',
            'yolo_decoder.module_list.22.BatchNorm2d.weight':'module_list.106.BatchNorm2d.weight',
            'yolo_decoder.module_list.22.BatchNorm2d.bias':  'module_list.106.BatchNorm2d.bias',
            'yolo_decoder.module_list.23.Conv2d.weight':     'module_list.107.Conv2d.weight',
            'yolo_decoder.module_list.23.BatchNorm2d.weight':'module_list.107.BatchNorm2d.weight',
            'yolo_decoder.module_list.23.BatchNorm2d.bias':  'module_list.107.BatchNorm2d.bias',
            'yolo_decoder.module_list.24.Conv2d.weight':     'module_list.108.Conv2d.weight',
            'yolo_decoder.module_list.24.BatchNorm2d.weight':'module_list.108.BatchNorm2d.weight',
            'yolo_decoder.module_list.24.BatchNorm2d.bias':  'module_list.108.BatchNorm2d.bias',
            'yolo_decoder.module_list.25.Conv2d.weight':     'module_list.109.Conv2d.weight',
            'yolo_decoder.module_list.25.BatchNorm2d.weight':'module_list.109.BatchNorm2d.weight',
            'yolo_decoder.module_list.25.BatchNorm2d.bias':  'module_list.109.BatchNorm2d.bias',
            'yolo_decoder.module_list.26.Conv2d.weight':     'module_list.110.Conv2d.weight',
            'yolo_decoder.module_list.26.BatchNorm2d.weight':'module_list.110.BatchNorm2d.weight',
            'yolo_decoder.module_list.26.BatchNorm2d.bias':  'module_list.110.BatchNorm2d.bias',
            'yolo_decoder.module_list.27.Conv2d.weight':     'module_list.111.Conv2d.weight',
            'yolo_decoder.module_list.27.BatchNorm2d.weight':'module_list.111.BatchNorm2d.weight',
            'yolo_decoder.module_list.27.BatchNorm2d.bias':  'module_list.111.BatchNorm2d.bias',
            'yolo_decoder.module_list.28.Conv2d.weight':     'module_list.112.Conv2d.weight',
            'yolo_decoder.module_list.28.Conv2d.bias':       'module_list.112.Conv2d.bias'
        }

        return mapping


    def load_weights(self, path):
        # These weights are original Yolo3 layers. Only load the decoder layers, do proper mappings
        pass

    def info(self, verbose=False):
        model_info(self, verbose)


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)
        # torch_utils.initialize_weights(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array([0, 2, 5], dtype=np.int32)  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info(verbose) if not ONNX_EXPORT else None  # print model description


    def forward(self, *xs):
        # Forward function from yolo3 without augmentation
        x = xs[0]
        e2 = xs[1]  # Replacing input from layer 36
        e3 = xs[2]  # Replacing input from layer 61

        encoder_inps = {36: e2, 61: e3}
        yolo_out, out = [], []

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['FeatureConcat']:  # concat
                x = module(x, out, encoder_inps)  # FeatureConcat()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            # out.append(x if self.routs[i] else [])
            out.append(x if i in self.routs else [])

        if self.training:  # train
            return yolo_out

        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs
            return x, p


    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        print('Fusing layers...')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info() if not ONNX_EXPORT else None  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False):
        model_info(self, verbose)


    def yolo3_layer_weight_mappings(self):
        mapping = {
            'yolo_decoder.module_list.0.Conv2d.weight':      'module_list.84.Conv2d.weight',
            'yolo_decoder.module_list.0.BatchNorm2d.weight': 'module_list.84.BatchNorm2d.weight',
            'yolo_decoder.module_list.0.BatchNorm2d.bias':   'module_list.84.BatchNorm2d.bias',
            'yolo_decoder.module_list.1.Conv2d.weight':      'module_list.85.Conv2d.weight',
            'yolo_decoder.module_list.1.BatchNorm2d.weight': 'module_list.85.BatchNorm2d.weight',
            'yolo_decoder.module_list.1.BatchNorm2d.bias':   'module_list.85.BatchNorm2d.bias',
            'yolo_decoder.module_list.2.Conv2d.weight':      'module_list.86.Conv2d.weight',
            'yolo_decoder.module_list.2.BatchNorm2d.weight': 'module_list.86.BatchNorm2d.weight',
            'yolo_decoder.module_list.2.BatchNorm2d.bias':   'module_list.86.BatchNorm2d.bias',
            'yolo_decoder.module_list.3.Conv2d.weight':      'module_list.87.Conv2d.weight',
            'yolo_decoder.module_list.3.BatchNorm2d.weight': 'module_list.87.BatchNorm2d.weight',
            'yolo_decoder.module_list.3.BatchNorm2d.bias':   'module_list.87.BatchNorm2d.bias',
            'yolo_decoder.module_list.4.Conv2d.weight':      'module_list.88.Conv2d.weight',
            'yolo_decoder.module_list.4.Conv2d.bias':        'module_list.88.Conv2d.bias',
            'yolo_decoder.module_list.7.Conv2d.weight':      'module_list.91.Conv2d.weight',
            'yolo_decoder.module_list.7.BatchNorm2d.weight': 'module_list.91.BatchNorm2d.weight',
            'yolo_decoder.module_list.7.BatchNorm2d.bias':   'module_list.91.BatchNorm2d.bias',
            'yolo_decoder.module_list.10.Conv2d.weight':     'module_list.94.Conv2d.weight',
            'yolo_decoder.module_list.10.BatchNorm2d.weight':'module_list.94.BatchNorm2d.weight',
            'yolo_decoder.module_list.10.BatchNorm2d.bias':  'module_list.94.BatchNorm2d.bias',
            'yolo_decoder.module_list.11.Conv2d.weight':     'module_list.95.Conv2d.weight',
            'yolo_decoder.module_list.11.BatchNorm2d.weight':'module_list.95.BatchNorm2d.weight',
            'yolo_decoder.module_list.11.BatchNorm2d.bias':  'module_list.95.BatchNorm2d.bias',
            'yolo_decoder.module_list.12.Conv2d.weight':     'module_list.96.Conv2d.weight',
            'yolo_decoder.module_list.12.BatchNorm2d.weight':'module_list.96.BatchNorm2d.weight',
            'yolo_decoder.module_list.12.BatchNorm2d.bias':  'module_list.96.BatchNorm2d.bias',
            'yolo_decoder.module_list.13.Conv2d.weight':     'module_list.97.Conv2d.weight',
            'yolo_decoder.module_list.13.BatchNorm2d.weight':'module_list.97.BatchNorm2d.weight',
            'yolo_decoder.module_list.13.BatchNorm2d.bias':  'module_list.97.BatchNorm2d.bias',
            'yolo_decoder.module_list.14.Conv2d.weight':     'module_list.98.Conv2d.weight',
            'yolo_decoder.module_list.14.BatchNorm2d.weight':'module_list.98.BatchNorm2d.weight',
            'yolo_decoder.module_list.14.BatchNorm2d.bias':  'module_list.98.BatchNorm2d.bias',
            'yolo_decoder.module_list.15.Conv2d.weight':     'module_list.99.Conv2d.weight',
            'yolo_decoder.module_list.15.BatchNorm2d.weight':'module_list.99.BatchNorm2d.weight',
            'yolo_decoder.module_list.15.BatchNorm2d.bias':  'module_list.99.BatchNorm2d.bias',
            'yolo_decoder.module_list.16.Conv2d.weight':     'module_list.100.Conv2d.weight',
            'yolo_decoder.module_list.16.Conv2d.bias':       'module_list.100.Conv2d.bias',
            'yolo_decoder.module_list.19.Conv2d.weight':     'module_list.103.Conv2d.weight',
            'yolo_decoder.module_list.19.BatchNorm2d.weight':'module_list.103.BatchNorm2d.weight',
            'yolo_decoder.module_list.19.BatchNorm2d.bias':  'module_list.103.BatchNorm2d.bias',
            'yolo_decoder.module_list.22.Conv2d.weight':     'module_list.106.Conv2d.weight',
            'yolo_decoder.module_list.22.BatchNorm2d.weight':'module_list.106.BatchNorm2d.weight',
            'yolo_decoder.module_list.22.BatchNorm2d.bias':  'module_list.106.BatchNorm2d.bias',
            'yolo_decoder.module_list.23.Conv2d.weight':     'module_list.107.Conv2d.weight',
            'yolo_decoder.module_list.23.BatchNorm2d.weight':'module_list.107.BatchNorm2d.weight',
            'yolo_decoder.module_list.23.BatchNorm2d.bias':  'module_list.107.BatchNorm2d.bias',
            'yolo_decoder.module_list.24.Conv2d.weight':     'module_list.108.Conv2d.weight',
            'yolo_decoder.module_list.24.BatchNorm2d.weight':'module_list.108.BatchNorm2d.weight',
            'yolo_decoder.module_list.24.BatchNorm2d.bias':  'module_list.108.BatchNorm2d.bias',
            'yolo_decoder.module_list.25.Conv2d.weight':     'module_list.109.Conv2d.weight',
            'yolo_decoder.module_list.25.BatchNorm2d.weight':'module_list.109.BatchNorm2d.weight',
            'yolo_decoder.module_list.25.BatchNorm2d.bias':  'module_list.109.BatchNorm2d.bias',
            'yolo_decoder.module_list.26.Conv2d.weight':     'module_list.110.Conv2d.weight',
            'yolo_decoder.module_list.26.BatchNorm2d.weight':'module_list.110.BatchNorm2d.weight',
            'yolo_decoder.module_list.26.BatchNorm2d.bias':  'module_list.110.BatchNorm2d.bias',
            'yolo_decoder.module_list.27.Conv2d.weight':     'module_list.111.Conv2d.weight',
            'yolo_decoder.module_list.27.BatchNorm2d.weight':'module_list.111.BatchNorm2d.weight',
            'yolo_decoder.module_list.27.BatchNorm2d.bias':  'module_list.111.BatchNorm2d.bias',
            'yolo_decoder.module_list.28.Conv2d.weight':     'module_list.112.Conv2d.weight',
            'yolo_decoder.module_list.28.Conv2d.bias':       'module_list.112.Conv2d.bias'
        }

        return mapping
