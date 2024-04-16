import numpy as np
import onnx
import torchvision.transforms as transforms
from onnx2pytorch import ConvertModel
from PIL import Image
from scipy.signal import convolve2d

transform = transforms.Compose(
    [
        transforms.Resize((358, 224)),
        transforms.ToTensor(),
    ]
)


onnx_model = onnx.load("end2end.onnx")
pytorch_model = ConvertModel(onnx_model)


def get_masked_image(image, threshold=0.3, mask_padding=5, bw="True"):
    torch_image = transform(image).unsqueeze(0)

    out = pytorch_model(torch_image)
    mask = np.where(out.squeeze(0, 1).detach().numpy() > threshold, 1, 0)

    kernel_size = 2 * mask_padding + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=int)

    mask = convolve2d(mask, kernel, mode="same", boundary="fill", fillvalue=0)

    image_sized = np.array(image.resize(mask.shape[::-1]))

    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    result = np.multiply(image_sized, mask)
    output_image = np.where(result == 0, 128, image_sized)

    if bw:
        return Image.fromarray(np.uint8(output_image)).convert("L")
    else:
        return Image.fromarray(np.uint8(output_image))
