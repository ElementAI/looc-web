from utils import transform_image, get_model


model = get_model()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    det = output.detach().cpu()
    result = int(det.sum().numpy())
    return result
