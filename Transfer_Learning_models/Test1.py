
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import pickle
plt.ion()   # interactive mode


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# imagenet_data = torchvision.datasets.ImageFolder('/home/dpw0002/Desktop/test2',data_transforms['test'])
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=4)
                        
#class_names = imagenet_data.classes
class_names = ( 'KM_Anbauteile','KM_Klappen_Frontklappen','KM_Klappen_Heckklappe','KM_Tueren','KM_Uebersichtsaufnahmen','KR_Anbauteile_Heckklappe','KR_Aufbau','KR_Aufbau_Heck','KR_Aufbau_Saeule_A','KR_Aufbau_Seitenteil')
num_classes=len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model=torch.load('Karosserie_trained.pth.tar')
loader=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(16,8))

    with torch.no_grad():
            image = Image.open('/home/dpw0002/Desktop/2_6_TW_IMG_0031.jpg')
            image = loader(image).float()
            image.unsqueeze_(0)
            #image = Variable(image, requires_grad=True)

        #for i, (inputs, labels) in enumerate(data_loader):
            inputs = image.to(device)
            #labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            #for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot()
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds]))
            print(inputs.cpu().data.size())
            imshow(inputs.cpu().data[0])
                

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)

visualize_model(model)

plt.ioff()
plt.show()