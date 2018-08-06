
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
testset = torchvision.datasets.ImageFolder(root='/home/dpw0002/Desktop/test_elek_FA', transform=data_transforms['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
class_names = testset.classes
#class_names = ('Abgasanlage','AF_Lenkung','AF_Motorraum_Anbauteile','Aggregate_Fahrwerk','Bremssystem','Getriebe','HA','Motorraum','VA')
num_classes=len(class_names)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model=torch.load('Fahrwerk_aggregate_trained.pth.tar')
loader=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
    fig = plt.figure(figsize=(16,8))

    with torch.no_grad():
            image = Image.open('/home/dpw0002/Desktop/4_6_TW_IMG_2037.jpg')
            image = loader(image).float()
            image.unsqueeze_(0)
            #image = Variable(image, requires_grad=True)

        #for i, (inputs, labels) in enumerate(data_loader):
            inputs = image.to(device)
            #labels = labels.to(device)
           
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            result = torch.topk(outputs.data[0], k=4)
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

            #for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot()
            ax.axis('off')
            #ax.set_title('predicted: {}'.format(class_names[preds]))
            print(inputs.cpu().data.size())
            imshow(inputs.cpu().data[0])
            for label, prob in result:
                print('label:%s,probability:%.4f'%(label, prob))    

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)

visualize_model(model)
#accuracy testing on each classes ''' Remember one point-When choosing the test dataset no.of images/4 or else it commits the index error.
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

# # with torch.no_grad():
# #     for data in testloader:
# #         images, labels = data
# #         outputs = model(images)
# #         _, predicted = torch.max(outputs.data, 1)
# #         total += labels.size(0)
# #         correct += (predicted == labels).sum().item()

# # print('Accuracy of the network on the 10000 test images: %d %%' % (
# #     100 * correct / total))


# class_correct = list(0. for i in range(9))
# class_total = list(0. for i in range(9))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images=images.to(device)
#         labels=labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1


# for i in range(9):
#     print('Accuracy of %5s : %2d %%' % (
#         class_names[i], 100 * class_correct[i] / class_total[i]))


plt.ioff()
plt.show()