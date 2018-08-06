
#tensorboard --logdir='./logs' --port=6006

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
from logger import Logger
import tensorflow as tf
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
#data_dir='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Easy/Exact_dataset'
data_dir='/media/dpw0002/740F759C1A78BC9F/Desktop_backup/Easy/Exact_dataset'
#data_dir='/home/dpw0002/Desktop/hymenoptera_data'


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=8)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


""" Tensonboard logs """
logger = Logger('./logs')
writer_val = tf.summary.FileWriter('./logs/plot_val')
writer_train = tf.summary.FileWriter('./logs/plot_train')
loss_var = tf.Variable(0.0)
tf.summary.scalar("loss", loss_var)
write_op = tf.summary.merge_all()
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^

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


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])
######################################################################
#Tensorboard logging


def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)  

def tensorboard_logging(phase,epoch_loss,epoch_acc,epoch,inputs):
    if phase =='train':
        info = {'train loss': epoch_loss, 'train accuracy': epoch_acc}
    else:
        info = {'val loss': epoch_loss, 'val accuracy': epoch_acc}
    # loss Train
    if phase=='train':
        summary = session.run(write_op, {loss_var: epoch_loss})
        writer_train.add_summary(summary, epoch+1)
        writer_train.flush()        
    else:
        # loss Validation
        summary = session.run(write_op, {loss_var: epoch_loss})
        writer_val.add_summary(summary, epoch+1)
        writer_val.flush()    



    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in model_ft.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), epoch+1)
        logger.histo_summary(tag+'/grad', to_np(value.grad), epoch+1)

    # (3) Log the images
    info = { 'images': to_np(inputs.view(-1, 224, 224)[:4])}

    for tag, images in info.items():
        logger.image_summary(tag, images, epoch+1)
######################################################################
# Training the model


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    #train and val scores figures
    fig=plt.figure()
    epochs=[]
    epoch_losses_train=[]
    epoch_losses_val=[]
    epoch_accuracy_train=[]
    epoch_accuracy_val=[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            tensorboard_logging(phase,epoch_loss,epoch_acc,epoch,inputs)
            if phase == 'train':
                epochs.append(epoch)
                epoch_losses_train.append(epoch_loss)
                epoch_accuracy_train.append(epoch_acc)
            else:
                #epochs.append(epoch)
                epoch_losses_val.append(epoch_loss)
                epoch_accuracy_val.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    #plot the results 
    plt.plot(epochs,epoch_losses_train,'b-',label='train Loss')
    plt.plot(epochs,epoch_losses_val,'r-',label='Validation Loss')
    plt.plot(epochs,epoch_accuracy_train,'g-',label='Training Accuracy')
    plt.plot(epochs,epoch_accuracy_val,'y-',label='Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('Loss and Accuracy')
    plt.title('RESNET-Final training')
    plt.grid(True)
    plt.show()
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model,'final_training.pth.tar')
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(16,8))

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

######################################################################
# Finetuning the convnet
# ----------------------

dclasses=len(class_names)
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs,dclasses)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001,momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)

######################################################################
#

visualize_model(model_ft)


######################################################################
# ConvNet as fixed feature extractor
# ----------------------------------


model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, dclasses)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate


# model_conv = train_model(model_conv, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=25)

# ######################################################################
# #

# visualize_model(model_conv)

plt.ioff()
plt.show()
