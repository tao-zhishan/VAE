# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:57:17 2021

@author: zhishan.tao
"""

import pickle
import numpy as np
from torch.utils.data.dataloader import DataLoader
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from  torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid as make_image_grid
from scipy.stats import norm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def read_freyfaces(path):
    fp = open(path, 'rb')
    data = pickle.load(fp, encoding='latin')
    return data



class VAE(nn.Module):
    def __init__(self,input_dim=560,latent_dim=20,hidden_dim=500):
        super(VAE,self).__init__()
        self.fc_e = nn.Linear(input_dim,hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim,latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim,latent_dim)
        self.fc_d1 = nn.Linear(latent_dim,hidden_dim)
        self.fc_d21 = nn.Linear(hidden_dim,input_dim) #x_mu
        self.fc_d22 = nn.Linear(hidden_dim,input_dim) #x_logsigma
        self.input_dim = input_dim
        
    def encoder(self,x_in):
        x = F.relu(self.fc_e(x_in.view(-1,self.input_dim)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
    def decoder(self,z):
        z = F.relu(self.fc_d1(z))
        x_out_mu = torch.sigmoid(self.fc_d21(z))
        x_out_logvar = torch.sigmoid(self.fc_d22(z))
        return x_out_mu,x_out_logvar
    
    def sample_normal(self,mean,logvar):
        sd = torch.exp(logvar*0.5)
        e = Variable(torch.randn(sd.size())) # Sample from standard normal
        z = e.mul(sd).add_(mean)
        return z
    
    def forward(self,x_in):
        z_mu, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mu,z_logvar)
        x_out_mu,x_out_logvar = self.decoder(z)
        return x_out_mu, x_out_logvar, z_mu, z_logvar


def freyfaces_criterion(x_out_mu,x_out_logvar,x_in,z_mu,z_logvar):
    # 重构loss为似然函数
    # -n/2(ln2pi + lnsigma^2 + (xi - mu)^2/sigma^2)
    recon_loss_ele = -0.5 *(np.log(np.pi) + x_out_logvar + 
                        ((x_in - x_out_mu).pow(2)/ torch.exp(x_out_logvar)))
    recon_loss = -torch.sum(recon_loss_ele)
    
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (recon_loss + kld_loss) / x_out_mu.size(0) # normalize by batch size
    return loss


def train(model,optimizer,dataloader,epochs=6):
    losses = []
    for epoch in range(epochs):
        for images in dataloader:
            x_in = Variable(images) #有15-20个x_in，一个x_in有100张图片
            optimizer.zero_grad()
            x_out_mu,x_out_logvar, z_mu, z_logvar = model(x_in)
            loss = freyfaces_criterion(x_out_mu,x_out_logvar,x_in,z_mu,z_logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        if epoch % 5 ==0:
            print ('====> Epoch:{} loss: {:.4f}'.format(
                epoch,
                loss))
    return losses

data = read_freyfaces("./data/freyfaces.pkl")
data = torch.FloatTensor(data)
train_x = data[:1500]
validation_x = data[1500:1700]
test_x = data[1700:]
size_h = 28
size_w = 20

batch_size = 100
train_loader = DataLoader(train_x,batch_size=batch_size)
model = VAE(input_dim=560, latent_dim=20, hidden_dim=200)
optimizer = torch.optim.Adam(model.parameters())

train_losses = train(model,optimizer,train_loader,epochs=20)
plt.figure(figsize=(10,5))
plt.plot(train_losses)
plt.show()


def visualize_freyface_vae(model,original_data,num=16):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1,2,0)))
        plt.axis('off')
        plt.show()
        
    images = original_data
    print(images.shape)
    images = images[0:num,:]
    x_in = Variable(images)
    print(images.shape)
    x_out,_,_,_ = model(x_in)
    x_out = x_out.data
    print(x_out.shape)
    imshow(make_image_grid(1-images.reshape(num,1,28,20)))
    imshow(make_image_grid(1-x_out.reshape(num,1,28,20)))

visualize_freyface_vae(model,test_x)



'''二维Z隐变量，解码图片'''

model2 = VAE(latent_dim=2)
optimizer = torch.optim.Adam(model2.parameters())
train_losses = train(model=model2,
                     optimizer=optimizer,
                     dataloader=train_loader,
                     epochs=200)
plt.Figure()
plt.plot(train_losses)
plt.show()

def visualize_decoder(model,num=20,range_type='g'):
    image_grid = np.zeros([num*28,num*20])

    if range_type == 'l': # linear range
        # corresponds to output range of visualize_encoding()
        range_space = np.linspace(-4,4,num)
    elif range_type == 'g': # gaussian range
        range_space = norm.ppf(np.linspace(0.01,0.99,num))
    else:
        range_space = range_type

    for i, x in enumerate(range_space):
        for j, y in enumerate(reversed(range_space)):
            z = Variable(torch.FloatTensor([[x,y]]))
            image,_ = model.decoder(z)
            image = image.data.numpy().reshape(28,20)
            image_grid[(j*28):((j+1)*28),(i*20):((i+1)*20)] = image

    plt.figure(figsize=(14, 10))
    plt.imshow(255-image_grid,cmap='Greys_r')
    plt.show()

visualize_decoder(model2,num=10,range_type='g')
