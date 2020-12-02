
import os
import glob
import torch
import cv2
from models.midas_net import MidasNet
from torchvision.transforms import Compose
from transforms.transforms import Resize, PrepareForNet, NormalizeImage
from utils.midas_utils import read_image, write_depth
from models.midasyolo3 import MidasYoloNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
print(device)


def run_only_midas(input_path, output_path, model_path, optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input folder
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model = MidasNet(model_path, non_negative=True)
    # net_w, net_h = 384, 384
    net_w, net_h = 416, 416

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True:
        rand_example = torch.rand(1, 3, net_h, net_w)
        model(rand_example)
        traced_script_module = torch.jit.trace(model, rand_example)
        model = traced_script_module

        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

    model.to(device)

    # get input
    img_names = glob.glob(os.path.join(input_path, "*"))
    num_images = len(img_names)

    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

    for ind, img_name in enumerate(img_names):

        print("  processing {} ({}/{})".format(img_name, ind + 1, num_images))

        # input

        img = read_image(img_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                    .squeeze()
                    .cpu()
                    .numpy()
            )

        # ToDo Remove this......
        # break

        # output
        filename = os.path.join(
            output_path, os.path.splitext(os.path.basename(img_name))[0]
        )
        write_depth(filename, prediction, bits=2)

        # # Delete the processed file from input folder -- TEST
        # os.remove(img_name)

    print("finished")

if __name__ == "__main__":
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    print(device)


    # ## Testing only midas
    # run_only_midas('input', 'output', 'model-f6b98070.pt')

    # ## Testing only yolo
    # cfg = "cfg/yolov3-custom.cfg"
    # yolo_model = Darknet(cfg).to(device)
    # yolo_model.info(True)

    ## Testing midas+yolo
    cfg = "cfg/yolov3-custom.cfg"
    midasyolo_model = MidasYoloNet(path='model-f6b98070.pt',  yolo_cfg=cfg)

    print("MIDAS YOLO COMBINED MODEL ========================")
    print(midasyolo_model)
    print("END ========================")


    # ## Testing DataLoader
    # data = LoadImagesAndLabels(path= "data/customdata/testfile.txt", cache_images=True)
    # print(data.img_files)
    # print(data.labels)
    # cv2.imshow('', data.imgs[0])
    # cv2.waitKey(0)