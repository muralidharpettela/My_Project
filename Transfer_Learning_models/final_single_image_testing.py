
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
import torch.nn.functional as F
import pickle
from matplotlib import rcParams
import matplotlib as mpl
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

testset = torchvision.datasets.ImageFolder(root='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Easy/Exact_dataset/val', transform=data_transforms['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
class_names = testset.classes

num_classes=len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model=torch.load('trained_models/final_training.pth.tar')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
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
    fig = plt.figure(figsize=(16,20))

    with torch.no_grad():
            image = Image.open('/home/dpw0002/Desktop/IMG_1455.jpg')
            image = data_transforms['test'](image).float()
            image.unsqueeze_(0)
            #image = Variable(image, requires_grad=True)

        #for i, (inputs, labels) in enumerate(data_loader):
            inputs = image.to(device)
            #labels = labels.to(device)
           
            outputs = model(inputs)
            output = F.softmax(outputs, dim=1)
            result = torch.topk(output.data[0], k=4)
            #_, preds = torch.max(outputs, 1)

            if device != "cpu":
                index = result[1].cpu().numpy().tolist()
                prob = result[0].cpu().numpy().tolist()
            else:
                index = result[1].numpy().tolist()
                prob = result[0].numpy().tolist()
            if class_names is not None:
                label = []
                for idx in index:
                    label.append(class_names[idx])
                result = zip(label, prob)
            else:
                result = zip(index, prob)

            for label, prob in result:
                print('label:%s,probability:%.4f'%(label, prob))  

            #for j in range(inputs.size()[0]):
            
            images_so_far += 1
            ax = plt.subplot()
            ax.axis('off')
            
            #for label, prob in result:
                #ax.set_title('predicted: \n label: {}, \n prob:'.format(label, prob))
                

            print(inputs.cpu().data.size())
            imshow(inputs.cpu().data[0])
              

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)

visualize_model(model)


plt.ioff()
plt.show()