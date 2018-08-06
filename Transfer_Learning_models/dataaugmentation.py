import torch
import torchvision
from PIL import Image
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

loader=transforms.Compose([
        transforms.TenCrop(180),
        transforms.Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])), # returns a 4D tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


image = Image.open('/home/dpw0002/Desktop/4_6_TW_IMG_2037.jpg')
image = loader(image).float()
image.unsqueeze_(0)

inputs = image.to(device)

imshow(inputs.cpu().data[0])

plt.ioff()
plt.show()