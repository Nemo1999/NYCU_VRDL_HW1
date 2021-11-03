from __future__ import print_function
import os
from PIL import Image

import tqdm
import time
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from progressbar import progressBar
from utils import *

import argparse
from datetime import datetime
import my_custom_dataset



def evaluate(model_path):

    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    print(use_cuda)

    # Data
    print('==> Preparing data..')

    transform_test = transforms.Compose([
        transforms.Scale((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])





    eval_set = my_custom_dataset.evalset(transform=transform_test)


    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=3)

    # Model

    net = torch.load(model_path)
    netp = torch.nn.DataParallel(net, device_ids=[0])

    # GPU
    device = torch.device("cuda")
    net.to(device)
    # evaluation mode
    net.eval()

    c2i, i2c = my_custom_dataset.get_class_dicts()
    with torch.no_grad():
        with open(os.path.join(os.path.dirname(model_path), f'eval_result{datetime.now().strftime("%m-%d-%Y__%H:%M:%S").replace(" ", "-")}.txt'), "a") as f:
            for batch_idx, (image_name, image) in tqdm.tqdm(enumerate(eval_loader)):
                idx = batch_idx
                if use_cuda:
                    image = image.to(device)
                image = Variable(image, volatile=True)
                output_1, output_2, output_3, output_concat= net(image)
                outputs_com = output_1 + output_2 + output_3 + output_concat

                _, predicted = torch.max(output_concat.data, 1)
                _, predicted_com = torch.max(outputs_com.data, 1) #this acc tends to be higher

                f.write(f'{image_name[0]} {i2c[predicted_com.item()]}\n')



    return


if __name__ == "__main__":

    #parse model path
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help='provide model path')
    args = parser.parse_args()
    # evaluate model
    evaluate(model_path=args.model_path)
