import io

import torch
import torchvision.transforms as transforms

from model import CSRNet

from PIL import Image


def get_model():
    model = CSRNet()
    model = model.cuda()
    model.eval()
    checkpoint = torch.load('2model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transforms(image).unsqueeze(0).cuda()
