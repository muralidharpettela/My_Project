from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
#torch

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
import torch.nn.functional as F

import json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

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

#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')

model=torch.load('/home/dpw0002/Desktop/Application/final_training.pth.tar')

def model_predict(img_path, model):
    image = Image.open(open(img_path,'rb'))
    image = data_transforms['test'](image).float()
    image.unsqueeze_(0)
    #image = Variable(image, requires_grad=True)

    #for i, (inputs, labels) in enumerate(data_loader):
    inputs = image.to(device)
    #labels = labels.to(device)
           
    outputs = model(inputs)
    #output = F.softmax(outputs, dim=1)
    #result = torch.topk(output.data[0], k=1)
    _, preds = torch.max(outputs, 1)
    data=class_names[preds]
    # if device != "cpu":
    #     index = result[1].cpu().numpy().tolist()
    #     prob = result[0].cpu().numpy().tolist()
    # else:
    #     index = result[1].numpy().tolist()
    #     prob = result[0].numpy().tolist()

    # if class_names is not None:
    #     label = []
    #     for idx in index:
    #         label.append(class_names[idx])
    #     results = zip(label, prob)
    # else:
    #     results = zip(index, prob)
    # data=[]
    # for label, prob in results:
    #     print('label:%s,probability:%.4f'%(label, prob)) 
    #     m_predict = {'label': label, 'probability': ("%.4f" % prob)}
    #     data.append(m_predict)

    return data

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)   
        #          # Simple argmax
        #data = list()
        #for label, prob in preds:
            #m_predict = {'label': label, 'probability': ("%.4f" % prob)}
            #data.append(m_predict)
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        #json.dumps(preds)
        return preds
        #return render_template("index.html", predict=preds)
    return None





if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

