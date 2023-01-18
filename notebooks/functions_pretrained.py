import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path


def get_image_data(data_dir, data_transforms, batch_size):
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "test"]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}

    return dataloaders, dataset_sizes


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    dataloaders,
    dataset_sizes,
    num_epochs,
):
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

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
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif phase == 'test':
                test_loss.append(epoch_loss)
                test_acc.append(epoch_acc)


            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, {'train_loss':train_loss, 'train_acc': train_acc, 'test_loss':test_loss, 'test_acc':test_acc}


def finetune_conv_net(
    model_ft, device, data_dir, data_transforms, num_epochs, batch_size=None
):

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    dataloaders, dataset_sizes = get_image_data(data_dir, data_transforms, batch_size)

    model_ft, tracking = train_model(
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        device,
        dataloaders,
        dataset_sizes,
        num_epochs=num_epochs,
    )

    return model_ft, tracking


def save_outputs(tracking, model, usage_type, num_epochs, batch_size):
    #Save loss and accuracy
    text_path = Path("..", "outputs", f'bwbatchsize_{batch_size}', model, f'{model}_{usage_type}_{num_epochs}epochs_{batch_size}batch_loss_accuracy.txt')

    with open(text_path, 'w') as data:
        data.write(str(tracking))

    #Plot Graphs
    epochs_range = range(len(tracking['train_acc']))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, tracking['train_acc'], label="Training Accuracy")
    plt.plot(epochs_range, tracking['test_acc'], label="Test Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Test Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, tracking['train_loss'], label="Training Loss")
    plt.plot(epochs_range, tracking['test_loss'], label="Test Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Test Loss")

    file_name = f'{model}_{usage_type}_{num_epochs}epochs_{batch_size}batch_graphs.png'
    graph_path = Path("..", "outputs", f'batchsize_{batch_size}', model, file_name)

    plt.savefig(graph_path, bbox_inches='tight')

def run_pretrained_nn(model, model_name, device, data_dir, data_transforms, num_epochs, batch_size, return_model=False):
    #Finetune conv net
    print('Finetuning Conv Net')
    fintuned_model, tracking_finetune = finetune_conv_net(
        model, device, data_dir, data_transforms, num_epochs=num_epochs, batch_size=batch_size
    )
    if return_model == False:
        save_outputs(tracking_finetune, model_name, 'finetuning', num_epochs, batch_size)

    #Run NN as a feature extractor
    #print('Run neural network as feature extractor')
    #feature_extractor_model, tracking_extractor = run_as_feature_extractor(
        #model, device, data_dir, data_transforms, num_epochs=num_epochs, batch_size=batch_size
    #)
    #save_outputs(tracking_extractor, model_name, 'feature_extractor', num_epochs, batch_size)

    if return_model == True:
        return fintuned_model

