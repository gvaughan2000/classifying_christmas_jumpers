# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode

# -

from functions_pretrained import get_image_data, finetune_conv_net, save_outputs, run_pretrained_nn

# +
BATCH_SIZE = 32
NUM_EPOCHS = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# -

# ## Set up

# +
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../data/5_BW_Train_Test_Folder'
# -

# # ResNet

# +
#resnet_18 = models.resnet18(pretrained=True)
    
#num_ftrs = resnet_18.fc.in_features
#resnet_18.fc = nn.Linear(num_ftrs, 2)

#run_pretrained_nn(resnet_18, 'test_resnet', device, data_dir, data_transforms, NUM_EPOCHS, BATCH_SIZE)
# -

# # Alex net

# +
alex_net = models.alexnet(pretrained=True)

alex_net.classifier[6] = nn.Linear(4096, 2)

run_pretrained_nn(alex_net, 'alexnet', device, data_dir, data_transforms, NUM_EPOCHS, BATCH_SIZE)
# -

# # VGG

# +
vgg = models.vgg16(pretrained=True)

vgg.classifier[6] = nn.Linear(4096, 2)

run_pretrained_nn(vgg, 'vgg', device, data_dir, data_transforms, NUM_EPOCHS, BATCH_SIZE)
# -

# # GoogLeNet

# +
googlenet = models.googlenet(pretrained=True)
googlenet.fc = nn.Linear(1024, 2)

run_pretrained_nn(googlenet, 'googlenet', device, data_dir, data_transforms, NUM_EPOCHS, BATCH_SIZE)

# +
googlenet = models.googlenet(pretrained=True)
googlenet.fc = nn.Linear(1024, 2)

final_model = run_pretrained_nn(googlenet, 'googlenet', device, data_dir, data_transforms, NUM_EPOCHS, BATCH_SIZE, return_model=True)
# -

# # Final Validation

# +
from pathlib import Path
import torchvision.transforms as T
from PIL import Image

transform_errors = T.ToPILImage()

running_corrects = 0

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

validation_data_path = Path('..', 'data', '5_BW_Train_Test_Folder', 'val')

image_dataset = datasets.ImageFolder(validation_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]))

dataloader = torch.utils.data.DataLoader(
            image_dataset, batch_size=1, shuffle=True, num_workers=4)

# +
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = final_model(inputs)
    _, preds = torch.max(outputs, 1)
    
    #If predicted correctly:
    if preds == labels.data:
        #add to correct
        running_corrects += 1
        if preds==1:
            true_positives +=1
        else:
            true_negatives +=1
        
    # If not prediced correctly
    else:
        #visualise image
        print(preds, labels.data)
        img = transform_errors(inputs[0])
        plt.imshow(img)
        plt.show()
        #img.save(Path('..', 'outptus', 'incorrect_on_validation'))
        
        if preds==1:
            false_positives+=1
        else:
            false_negatives+=1

    print(running_corrects)
    
    
final_accuracy = (running_corrects / len(dataloader))*100

print(final_accuracy, 'final_accuracy')
print('true_positives ', true_positives)
print('true_negatives ', true_negatives)
print('false_positives ', false_positives)
print('false_negatives ', false_negatives)

print('(1) = normal, (0)=christmsas')
