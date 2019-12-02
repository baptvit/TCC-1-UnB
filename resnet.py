import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import csv
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision import datasets, models, transforms
from __future__ import print_function 
from __future__ import division
from torch.optim import lr_scheduler

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir='/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final/dataset_final'

MODEL_SAVE_DIR='/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final'

CSV_TRAIN_DIR='/content/gdrive/My Drive/UnB/TCC-1/TCC1-1-dataset-final'

# If you dont have a pre-trained model or want to recovery traing
# let this field None or False
PREVIUS_TRAINED_MODEL=False

# Models to choose from [resnet]
model_name='resnet'

# Number of classes in the dataset
num_classes=9

# Batch size for training (change depending on how much memory you have)
batch_size=32

# Number of epochs to train for 
num_epochs=100

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract=False

# Hyperparamets
STEP_SIZE_CONST=1

GAMMA_CONST=0.1 

LR_CONST=0.001

MOMENTUM_COST=0.9

def train_model(model, dataloaders, criterion, scheduler, optimizer, num_epochs=25, is_inception=False):
'''
The train_model function handles the training and validation of a given model. As input, it takes a PyTorch model,
 a dictionary of dataloaders, a loss function, an optimizer, a specified number of epochs to train and validate for,
  and a boolean flag for when the model is an Inception model. The is_inception flag is used to accomodate the
  Inception v3 model, as that architecture uses an auxiliary output and the overall model loss respects both the
  auxiliary output and the final output, as
  described here <https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958>__.
  The function trains for the specified number of epochs and after each epoch runs a full validation step. It also
  keeps track of the best performing model (in terms of validation accuracy), and at the end of training returns the
  best performing model. After each epoch, the training and validation accuracies are printed.
'''
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        since_epoch = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            time_elapsed_epoch = time.time() - since_epoch
            print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # Write in csv training infos
            row = [phase, epoch_loss, epoch_acc]
            with open(CSV_TRAIN_DIR + '/train_val_phase.csv', 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
                                                             
            csvFile.close()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                model.class_to_idx = image_datasets['train'].class_to_idx                                             
                state = {
                      'epoch': epoch,
                      'arch': 'resnet152',
                      'state_dict': model.state_dict(),
                      'class_to_idx': model.class_to_idx,
                      'optimizer': optimizer.state_dict(),
                      }

                torch.save(state, MODEL_SAVE_DIR + '/restnet_model152_trained_exp7.pt')
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def set_parameter_requires_grad(model, feature_extracting):
'''
This helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature
extracting. By default, when we load a pretrained model all of the parameters have .requires_grad=True, which is fine
if we are training from scratch or finetuning. However, if we are feature extracting and only want to compute
gradients for the newly initialized layer then we want all of the other parameters to not require gradients.
This will make more sense later.
'''
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 448

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224


    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True, num_workers=16) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'val']}

class_names = image_datasets['train'].classes

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(dataset_sizes)

# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.

params_to_update = model_ft.parameters()

print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=LR_CONST, momentum=MOMENTUM_COST)

optimizer_ft = optim.SGD(params_to_update, lr=0.01, weight_decay=0.00001, momentum=MOMENTUM_COST)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=STEP_SIZE_CONST, gamma=GAMMA_CONST)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model_ft.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# Should result in 
# 58,162,249 total parameters.
# 58,162,249 training parameters.


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, exp_lr_scheduler,
                             optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
