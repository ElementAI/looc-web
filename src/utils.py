import io
import importlib

from pathlib import Path
from typing import BinaryIO

import yaml
import torch
import torchvision.transforms as transforms

from PIL import Image


with open('config.yml') as conf:
    configuration = yaml.safe_load(conf)

local_path = Path.cwd()
model_name = configuration["model"]["architecture"]
models_path = local_path / 'models' / model_name
models_path = models_path.with_suffix('.py')


def get_model():
    '''Instantiate and prepare the model
    Load checkpoint'''
    class_name = model_name
    target_class = dynamic_import(models_path, class_name, model_name)
    model = target_class()
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(configuration["model"]["checkpoint"])
    model.load_state_dict(checkpoint['state_dict'])
    return model


def transform_image(image_bytes: BinaryIO) -> torch.Tensor:
    '''
    Apply transform to image before passing on to the model

    Args:
        image_bytes: binary image to be processed

    Returns:
        Image tensor after appliying transforms
    '''
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0).cuda()


def dynamic_import(abs_module_path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, abs_module_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    target_class = getattr(model, class_name)
    return target_class
