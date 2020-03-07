from typing import BinaryIO

from utils import transform_image, get_model


model = get_model()


def get_prediction(image_bytes: BinaryIO) -> int:
    '''Get an image binary as input, apply the required transforms,
    pass returned tensor through the model and return the sum of the
    resulting value'''
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    det = output.detach().cpu()
    result = int(det.sum().numpy())
    return result
