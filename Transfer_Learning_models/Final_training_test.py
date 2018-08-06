#
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
}

#data_dir = 'C:\\Users\\pettm\\Desktop\\Richtige_Dataset_Sauber'
#data_dir = '/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Karosserie'
#data_dir='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Karosserie'
#data_dir = '/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Fahrwerk'
#data_dir = '/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Elektrik'
data_dir='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Easy/Exact_dataset'
#data_dir='/home/dpw0002/Desktop/hymenoptera_data'
#data_dir='/home/dpw0002/Desktop/hymenoptera_data'
#data_dir='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Richtige_Dataset_Elektrik'


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes=len(class_names)

model=torch.load('Final_training.pth.tar')

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
    fig = plt.figure(figsize=(20,8))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                #ax.set_title('Ground truth: {}'.format(class_names[labels[j]]))
                ax.set_title('Ground truth: {}, predicted: {}'.format(class_names[labels[j]],class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model)

#accuracy testing on each classes ''' Remember one point-When choosing the test dataset no.of images/4=remainder(0) or else it commits the index error.
# correct = 0
# total = 0
# with torch.no_grad():

#     for i, (inputs, labels) in enumerate(testloader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)

#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (preds == labels).sum().item()
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))

# class_correct = list(0. for i in range(25))
# class_total = list(0. for i in range(25))
# with torch.no_grad():
#     for i, (inputs, labels) in enumerate(testloader):
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(25):
#     print('Accuracy of %5s : %2d %%' % (
#         class_names[i], 100 * class_correct[i] / class_total[i]))

plt.ioff()
plt.show()