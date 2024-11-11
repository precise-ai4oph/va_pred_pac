from typing import Union, Tuple, Optional, Dict, Any

import torch
import torchvision
import torchvision.transforms.functional as TF

from PIL import Image, ImageDraw

def get_norm_mean_std(model_name:str):
    if model_name.startswith('resnet'):
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif model_name.startswith('efficientnetv2'):
        mean = [0.485, 0.456, 0.406] if model_name.endswith('s') or model_name.endswith('s-pd')  else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if model_name.endswith('s') or model_name.endswith('s-pd') else [0.5, 0.5, 0.5]
        return mean, std
    else:
        raise ValueError("Wrong model_name: {}".format(model_name))


def get_transforms(model_name:str, pretrain: Union[None, str], **kwargs: Any) -> Tuple:
    if kwargs is not None:
        no_normalize = kwargs['no_normalize'] if 'no_normalize' in kwargs else False
    else:
        no_normalize = False

    def get_normalize_transforms(no_normalize, mean, std):
        if no_normalize: return []
        return [torchvision.transforms.Normalize(mean=mean, std=std)]

    if pretrain is None and not model_name.startswith('efficientnetv2'):
        train_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(512),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor()])
        transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(512),
                    torchvision.transforms.ToTensor()])
    elif model_name.startswith('resnet'):
        resize_size = 256 if pretrain == 'V1' else 232
        crop_size = 224
        mean, std = get_norm_mean_std(model_name)
        train_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.CenterCrop(crop_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation((-5,5)),
                    torchvision.transforms.ToTensor()]
                    + get_normalize_transforms(no_normalize, mean=mean, std=std))
        transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize_size, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
                    torchvision.transforms.CenterCrop(crop_size),
                    torchvision.transforms.ToTensor()]
                    + get_normalize_transforms(no_normalize, mean=mean, std=std))
    elif model_name.startswith('efficientnetv2'):
        resize_size = 384 if model_name.endswith('s') else 480
        crop_size = resize_size
        interpolation = torchvision.transforms.InterpolationMode.BILINEAR if model_name.endswith('s') else torchvision.transforms.InterpolationMode.BICUBIC
        mean, std = get_norm_mean_std(model_name)
        train_transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize_size, interpolation=interpolation),
                    torchvision.transforms.CenterCrop(crop_size),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomRotation((-10,10)),
                    torchvision.transforms.ToTensor()]
                    + get_normalize_transforms(no_normalize, mean=mean, std=std))
        transforms = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(resize_size, interpolation=interpolation),
                    torchvision.transforms.CenterCrop(crop_size),
                    torchvision.transforms.ToTensor()]
                    + get_normalize_transforms(no_normalize, mean=mean, std=std))
    elif model_name.startswith('lenet5') or model_name.startswith('simplecnn'):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation((-10, 10)),
            RandomGammaCorrection((0.8,1.2)),
            torchvision.transforms.ToTensor()])
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])
    return train_transforms, transforms

