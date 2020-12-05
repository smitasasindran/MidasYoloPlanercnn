import math
import torch
import torch.nn as nn

from models.layers import YOLOLayer, FeatureConcat, EmptyLayer
# Delete this, not required
ONNX_EXPORT = False


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)

    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand==True:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


# Yolo create modules function to create from config file
def create_modules(module_defs, img_size):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)

    # ToDo Smita: Make this configurable?
    # output_filters = [3]  # input channels
    output_filters = [2048]  # input channels from encoder

    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef['pad'] else 0,
                                                       groups=mdef['groups'] if 'groups' in mdef else 1,
                                                       bias=not bn))
            # else:  # multiple-size conv
            #     print("AAAAAA IN MIXEDCONV2D - Going once, going twice! AAA") # NEver called
            #     modules.add_module('MixConv2d', MixConv2d(in_ch=output_filters[-1],
            #                                               out_ch=filters,
            #                                               k=k,
            #                                               stride=stride,
            #                                               bias=not bn))

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)
                print("Adding to routes from convolutional:", i)

            if mdef['activation'] == 'leaky':  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
            # elif mdef['activation'] == 'swish':
            #     modules.add_module('activation', Swish())

        # elif mdef['type'] == 'BatchNorm2d':
        #     filters = output_filters[-1]
        #     modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4)
        #     if i == 0 and filters == 3:  # normalize RGB image
        #         # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
        #         modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
        #         modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))  # img_size = (320, 192)
            else:
                modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            # ToDo Smita: Hardcoding the channel sizes here.. coming from midas encoder
            encoder_inps = 256 # Pass it as features parameter
            filters = sum([encoder_inps if l > 0 else output_filters[l] for l in layers])
            # filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # print("In routes check 1: i=", i)
            # print("In routes check 2: layers=", layers)
            # print("In routes check 3: filters=", filters)

            routs.extend([i + l if l < 0 else l for l in layers])
            # print("In routes check 4: final routes=", routs)
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            # modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
            modules = EmptyLayer()

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [32, 16, 8, 4, 2][yolo_index]  # P3-P7 stride
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],  # anchor list
                                nc=mdef['classes'],  # number of classes
                                img_size=img_size,  # (416, 416)
                                yolo_index=yolo_index,  # 0, 1, 2...
                                layers=layers,  # output layers
                                stride=stride)

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                print('WARNING: smart bias initialization failure.')

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    # ToDo Smita: Changed routs_binary to dict, since layers are not present
    # routs_binary = [False] * (i + 1)
    routs_binary = {}
    # print("Routes=", routs)
    # print("Routs_binary=", routs_binary)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary
