#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch import nn, optim
from torch.nn import functional as F
# import dataloader from torch
# import dataset
# import toTesor
import torchvision
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import sys
from PIL import Image
import cv2
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

from dataset import TrainDatasetFromFolder
from model import Generator, Discriminator, GeneratorLoss
from tqdm import tqdm
from torch.autograd import Variable

# In[2]:


torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# In[3]:


UPSCALE_FACTOR = 8
CROP_SIZE = 128
N_EPOCHS = 35

# In[4]:


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# In[5]:


train_set = TrainDatasetFromFolder("data/original/train/", crop_size=CROP_SIZE,
                                   upscale_factor=UPSCALE_FACTOR, inter=Image.HAMMING)

trainloader = DataLoader(train_set, batch_size=64, num_workers=4, shuffle=True)

# In[6]:


netG = Generator(UPSCALE_FACTOR)
netD = Discriminator()

generator_criterion = GeneratorLoss()

generator_criterion = generator_criterion.to(device)
netG = netG.to(device)
netD = netD.to(device)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002)
optimizerD = optim.Adam(netD.parameters(), lr=0.0002)

# In[7]:


results = {
    "d_loss": [],
    "g_loss": [],
    "d_score": [],
    "g_score": []
}


# In[8]:


def train():
    global netG, netD, optimizerG, optimizerD, generator_criterion, N_EPOCHS, trainloader, results
    for epoch in range(1, N_EPOCHS + 1):
        train_bar = tqdm(trainloader)
        running_results = {'batch_sizes': 0, 'd_loss': 0,
                           "g_loss": 0, "d_score": 0, "g_score": 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            real_img = Variable(target)
            real_img = real_img.to(device)
            z = Variable(data)
            z = z.to(device)

            ## Update Discriminator ##
            fake_img = netG(z)
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ## Now update Generator
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            netG.zero_grad()
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += real_out.item() * batch_size

            ## Updating the progress bar
            train_bar.set_description(desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f" % (
                epoch, N_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']
            ))
            results["d_loss"].append(running_results['d_loss'] / running_results['batch_sizes'])
            results["g_loss"].append(running_results['g_loss'] / running_results['batch_sizes'])
            results["d_score"].append(running_results['d_score'] / running_results['batch_sizes'])
            results["g_score"].append(running_results['g_score'] / running_results['batch_sizes'])

        netG.eval()
    return results, netG, netD


# In[9]:


res, netG, netD = train()
# plot the loss from the  res
fig_1 = plt.figure(1)
plt.plot(res["d_loss"], label="Discriminator Loss")
plt.plot(res["g_loss"], label="Generator Loss")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")

#CHANGE HERE
fig_1.savefig("loss_graphs/loss_disgen_hamming_4_128_35.png")

fig_2 = plt.figure(2)
# plt.plot(res["d_loss"], label="Discriminator Loss")
plt.plot(res["g_loss"], label="Generator Loss")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")

#linx, bilinx, boxx, nnx, bicubicx, lanczos, hamming

#HERE AND HERE
fig_2.savefig("loss_graphs/hamming_4_128_35.png")
torch.save(netG.state_dict(), "good_models/hamming_4_128_35.pth")



# # Predicting

# In[10]:


# low_res, high_res = train_set[0]
#
# plt.imshow(low_res.permute(1, 2, 0))
# plt.show()
#
# # In[11]:
#
#
# # predict the outout using netD and netG
# predicted = netG(low_res.unsqueeze(0).to(device))
#
# plt.imshow(predicted.squeeze(0).detach().cpu().permute(1, 2, 0))
# plt.show()
#
# # In[34]:
#
#
# # select an image from data/original/train
#
# img = Image.open("data/original/train/image_0.jpg")
# # resize the image to 96x96
# img = img.resize((320, 160), Image.HAMMING)
#
# # convert to tensor
# img = ToTensor()(img)
#
# # predict
#
# img = torch.unsqueeze(img, 0)
# img = img.to(device)
# # predict model using  descriminator and generator
# predicted = netG(img)
#
# #  show the image
# print(predicted.shape)
#
# plt.imshow(predicted.squeeze(0).detach().cpu().permute(1, 2, 0))
# # plt.imshow(predicted.squeeze(0).detach().cpu().permute(1, 2,  0))
# plt.show()
#
# # In[36]:
#
#
# img = Image.open("data/original/train/image_0.jpg")
# # resize the image to 96x96
# img = img.resize((320, 160), Image.HAMMING)
#
# # plot the image
# plt.imshow(img)
# plt.show()
#
