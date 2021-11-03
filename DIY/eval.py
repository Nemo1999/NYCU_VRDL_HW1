import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as td
import torchvision as tv
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import sklearn.model_selection as skms
import sklearn.metrics as skmc
import my_custom_dataset
from datetime import datetime

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
num_classes = 201

#parse model path
parser=argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str, help='provide model path')
args = parser.parse_args()



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

# load eval images
ds_eval = my_custom_dataset.evalset(transform=transforms_eval)
test_loader = td.DataLoader(dataset=ds_eval, batch_size=1)

# instantiate the model
model = tv.models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, num_classes)
model = model.to(DEVICE)
print(model)

# load snapshot of the best model
model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))


# test the model
true = list()
pred = list()
c2i, i2c = my_custom_dataset.get_class_dicts()
model.eval()
with torch.no_grad():
    with open(f'result{datetime.now().strftime("%m-%d-%Y__%H:%M:%S").replace(" ", "-")}.txt', "a") as f:
        for batch in tqdm(test_loader):
            img_path, img = batch
            img = img.to(DEVICE)

            # predict bird species
            y_pred = model(img)
            #print(y_pred)
            pred = y_pred.argmax(dim=-1)
            #print(pred)
            f.write(f'{img_path[0]} {i2c[pred.item()]}\n')
