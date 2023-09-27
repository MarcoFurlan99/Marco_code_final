import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from UNet.utils.data_loading import BasicDataset
from UNet.unet import UNet
from UNet.unet_parameters import *

from tqdm import tqdm

def predict_img(net,
                full_img,
                device
                ):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, img_scale, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def predict(input, output, model, mask_threshold = 0.5, scale = 1.0, bilinear = False, classes = 2):

    in_files = input
    out_files = output

    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    for i, filename in enumerate(tqdm(in_files, disable = True)):
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           device=device)

        out_filename = out_files[i]
        result = mask_to_image(mask, mask_values)
        result.save(out_filename)
