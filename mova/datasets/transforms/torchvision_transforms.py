from torchvision import transforms

from mova.registry import TRANSFORMS


def register_torchvision_transforms():
    for cls_name, obj in transforms.__dict__.items():
        if cls_name[0].isupper():
            TRANSFORMS.register_module(name=f'TV{cls_name}', module=obj)

register_torchvision_transforms()
