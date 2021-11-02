import os
import sys
import tqdm
import numpy as np
import random
import csv
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sklearn.model_selection as skms
import sklearn.metrics as skmc
import my_custom_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = './Results'
RANDOM_SEED = 42
PRETRAINED = True

random.seed(RANDOM_SEED)

# create an output folder
os.makedirs(OUT_DIR, exist_ok=True)


# set hyper-parameters
params = {'batch_size': 24, 'num_workers': 12}
num_epochs = 100
num_classes = 201

def get_model_desc(pretrained=False, num_classes=200, use_attention=False):
    """
    Generates description string.
    """
    desc = list()

    if pretrained:
        desc.append('Transfer')
    else:
        desc.append('Baseline')

    if num_classes == 204:
        desc.append('Multitask')

    if use_attention:
        desc.append('Attention')

    return '-'.join(desc)


def log_accuracy(path_to_csv, desc, acc, sep='\t', newline='\n'):
    """
    Logs accuracy into a CSV-file.
    """
    file_exists = os.path.exists(path_to_csv)

    mode = 'a'
    if not file_exists:
        mode += '+'

    with open(path_to_csv, mode) as csv:
        if not file_exists:
            csv.write(f'setup{sep}accuracy{newline}')

        csv.write(f'{desc}{sep}{acc}{newline}')



# transform images

def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width).
    Fills up the padded area with value(s) passed to the `fill` parameter.
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad images to 500 pixels
max_padding = tv.transforms.Lambda(lambda x: pad(x, fill=fill))

transforms_train = tv.transforms.Compose([
   max_padding,
   tv.transforms.RandomOrder([
       tv.transforms.RandomCrop((375, 375)),
       tv.transforms.RandomHorizontalFlip(),
       tv.transforms.RandomVerticalFlip()
   ]),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_eval = tv.transforms.Compose([
   max_padding,
   tv.transforms.CenterCrop((375, 375)),
   tv.transforms.ToTensor(),
   tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# load images
ds_train = my_custom_dataset.trainset(transform=transforms_train)
ds_val = my_custom_dataset.testset(transform=transforms_train)
ds_eval = my_custom_dataset.evalset(transform=transforms_eval)

# instantiate data loaders
train_loader = td.DataLoader(
    dataset=ds_train,
    shuffle=True,
    **params
)
val_loader = td.DataLoader(
    dataset=ds_val,
    shuffle=True,
    **params
)
test_loader = td.DataLoader(dataset=ds_eval, batch_size=1)

# instantiate the model
model = tv.models.resnet50(pretrained=PRETRAINED)
model.fc = nn.Linear(2048, num_classes)
model = model.to(DEVICE)
print(model)

# instantiate optimizer and scheduler
if PRETRAINED :
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)



# generate model description string
model_desc = get_model_desc(num_classes=num_classes, pretrained=PRETRAINED)

# define the training loop
best_snapshot_path = None
val_acc_avg = list()
best_val_acc = -1.0

for epoch in range(num_epochs):

    # train the model
    model.train()
    train_loss = list()
    for batch in tqdm.tqdm(train_loader,postfix=({"epoch":int(epoch)})):
        x, y = batch

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        # predict bird species
        y_pred = model(x)

        # calculate the loss
        loss = F.cross_entropy(y_pred, y)

        # backprop & update weights
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

    # validate the model
    model.eval()
    val_loss = list()
    val_acc = list()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # predict bird species
            y_pred = model(x)

            # calculate the loss
            loss = F.cross_entropy(y_pred, y)

            # calculate the accuracy
            acc = skmc.accuracy_score([val.item() for val in y], [val.item() for val in y_pred.argmax(dim=-1)])

            val_loss.append(loss.item())
            val_acc.append(acc)

        val_acc_avg.append(np.mean(val_acc))

        # save the best model snapshot
        current_val_acc = val_acc_avg[-1]
        if current_val_acc > best_val_acc:
            if best_snapshot_path is not None:
                os.remove(best_snapshot_path)

            best_val_acc = current_val_acc
            best_snapshot_path = os.path.join(OUT_DIR, f'model_{model_desc}_ep={epoch}_acc={best_val_acc}.pt')

            torch.save(model.state_dict(), best_snapshot_path)

    # adjust the learning rate
    scheduler.step()

    # print performance metrics
    if (epoch == 0) or ((epoch + 1) % 1 == 0):
        sys.stdout.write('\rEpoch {} |> Train. loss: {:.4f} | Val. loss: {:.4f} | Val. acc: {:.4f}]\n'.format(
            epoch + 1, np.mean(train_loss), np.mean(val_loss), np.mean(val_acc),flush=True)
        )
