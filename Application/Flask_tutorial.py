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
plt.ion()
from flask import Flask, render_template, request
from werkzeug import secure_filename
import numpy as np
import cv2
import matplotlib.pyplot as plt, mpld3

app = Flask(__name__)


@app.route('/upload')
def upload_file():
   return render_template('fileupload.html')

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
    mpld3.show()
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      class_names = ('Elektrik_Antenna','Elektrik_FM_Wischeranlage','Elektrik_Frontend_Motoraum','Elektrik_Heckleuchten_Bremsleuchte','Elektrik_Kofferraum_Heckklappe','Elektrik_Scheinwerfer_Blinker')
      num_classes=len(class_names)
      #fig = plt.figure(figsize=(16,8))
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
      model=torch.load('trained_models/Elektrik_Clean.pth.tar')
      model.eval()
      with torch.no_grad():
            image=Image.open(open(f.filename,'rb'))
            image = loader(image).float()
            image.unsqueeze_(0)
            inputs = image.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            ax = plt.subplot()
            ax.axis('off') 
            ax.set_title('predicted: {}'.format(class_names[preds]))
            #print(inputs.cpu().data.size())
            imshow(inputs.cpu().data[0])
            mpld3.show()
      #return f
      return 'file uploaded successfully'


		
if __name__ == '__main__':
   app.run(debug=True, use_debugger=False, use_reloader=False)