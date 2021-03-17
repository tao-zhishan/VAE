import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.stats import norm
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import make_grid as make_image_grid
from tqdm import tnrange
from mpl_toolkits.mplot3d import Axes3D


class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20, hidden_dim=500):
        super(VAE, self).__init__()
        self.fc_e = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_d1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, input_dim)
        self.input_dim = input_dim

    def encoder(self, x_in):
        x = F.relu(self.fc_e(x_in.view(-1, self.input_dim)))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def decoder(self, z):
        z = F.relu(self.fc_d1(z))
        x_out = torch.sigmoid(self.fc_d2(z))
        return x_out.view(-1, 1, 28, 28)

    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn(sd.size()))  # Sample from standard normal
        z = e.mul(sd).add_(mean)
        return z

    def forward(self, x_in):
        z_mu, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mu, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mu, z_logvar


# Loss function
def criterion(x_out, x_in, z_mu, z_logvar):
    # BOUNOULLI分布交叉熵
    bce_loss = F.binary_cross_entropy(x_out, x_in, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + z_logvar - (z_mu ** 2) - torch.exp(z_logvar))
    loss = (bce_loss + kld_loss) / x_out.size(0)  # normalize by batch size
    return loss


def train(model, optimizer, dataloader, epochs=1):
    losses = []
    for epoch in range(epochs):
        for images, _ in dataloader:
            x_in = Variable(images)  # 有600个x_in，一个x_in有100张图片
            optimizer.zero_grad()
            x_out, z_mu, z_logvar = model(x_in)
            loss = criterion(x_out, x_in, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        if epoch % 5 == 0:
            print('====> Epoch:{} loss: {:.4f}'.format(
                epoch,
                loss))
    return losses


def test(model, dataloader):
    running_loss = 0.0
    for images, _ in dataloader:
        x_in = Variable(images)
        x_out, z_mu, z_logvar = model(x_in)
        loss = criterion(x_out, x_in, z_mu, z_logvar)
        running_loss = running_loss + (loss.data * x_in.size(0))
    return running_loss / len(dataloader.dataset)



model = VAE()
optimizer = torch.optim.Adam(model.parameters())
batch_size = 100
trainloader = DataLoader(
    MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)
testloader = DataLoader(
    MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

train_losses = train(model, optimizer, trainloader, epochs=10)
#plt.figure(figsize=(15, 5))
train_nums = np.array(range(len(train_losses)))*100 #100是batch数
plt.plot(train_nums,train_losses)
plt.show()


def visualize_losses_moving_average(losses,window=50,boundary='valid',ylim=(95,125)):
    mav_losses = np.convolve(losses,np.ones(window)/window,boundary)
    corrected_mav_losses = np.append(np.full(window-1,np.nan),mav_losses)
    plt.figure(figsize=(10,5))
    train_nums = np.array(range(len(losses)))*100
    #plt.plot(train_nums,losses)
    plt.plot(train_nums,corrected_mav_losses)
    plt.ylim(ylim)
    plt.show()
    
visualize_losses_moving_average(train_losses)

'''训练loss'''
test_loss = test(model, testloader)
print(test_loss)


# Visualize VAE input and reconstruction
def visualize_mnist_vae(model, dataloader, num=16):
    def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        plt.show()

    images, _ = iter(dataloader).next()
    images = images[0:num, :, :]
    x_in = Variable(images)
    x_out, _, _ = model(x_in)
    x_out = x_out.data
    imshow(make_image_grid(images))
    imshow(make_image_grid(x_out))


visualize_mnist_vae(model, testloader)

'''latent d = 2模型'''
model2 = VAE(latent_dim=2)
optimizer = torch.optim.Adam(model2.parameters())
train_losses = train(model=model2,
                     optimizer=optimizer,
                     dataloader=trainloader,
                     epochs=10)

visualize_losses_moving_average(train_losses)
'''隐变量可视化'''

def visualize_encoder(model, dataloader):
    z_means_x, z_means_y, all_labels = [], [], []

    for images, labels in iter(dataloader):
        z_means, _ = model.encoder(Variable(images))
        z_means_x = np.append(z_means_x, z_means[:, 0].data.numpy())
        z_means_y = np.append(z_means_y, z_means[:, 1].data.numpy())
        all_labels = np.append(all_labels, labels.numpy())

    plt.figure(figsize=(6.5, 5))
    plt.scatter(z_means_x, z_means_y, c=all_labels, cmap='inferno')
    plt.colorbar()
    plt.show()


visualize_encoder(model2, testloader)


# 隐变量space grid可视化数字图片
def visualize_decoder(model, num=20, range_type='g'):
    image_grid = np.zeros([num * 28, num * 28])

    if range_type == 'l':  # linear range
        # corresponds to output range of visualize_encoding()
        range_space = np.linspace(-4, 4, num)
    elif range_type == 'g':  # gaussian range
        range_space = norm.ppf(np.linspace(0.01, 0.99, num))
    else:
        range_space = range_type

    for i, x in enumerate(range_space):
        for j, y in enumerate(reversed(range_space)):
            z = Variable(torch.FloatTensor([[x, y]]))
            image = model.decoder(z)
            image = image.data.numpy()
            image_grid[(j * 28):((j + 1) * 28), (i * 28):((i + 1) * 28)] = image

    plt.figure(figsize=(10, 10))
    plt.imshow(image_grid, cmap='Greys_r')
    plt.show()


visualize_decoder(model2, num=15, range_type='g')

'''三维z散点图'''


def triD_visualize_encoder(model, dataloader):
    z1_means, z2_means, z3_means, all_labels = [], [], [], []

    for images, labels in iter(dataloader):
        z_means, _ = model.encoder(Variable(images))
        z1_means = np.append(z1_means, z_means[:, 0].data.numpy())
        z2_means = np.append(z2_means, z_means[:, 1].data.numpy())
        z3_means = np.append(z3_means, z_means[:, 2].data.numpy())
        all_labels = np.append(all_labels, labels.numpy())

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(z1_means, z2_means, z3_means, c=all_labels, cmap='inferno', label=all_labels)
    ax.set_zlabel('Z3', fontdict={'size': 10})
    ax.set_ylabel('Z2', fontdict={'size': 10})
    ax.set_xlabel('Z1', fontdict={'size': 10})
    plt.show()


model3 = VAE(latent_dim=3)
optimizer = torch.optim.Adam(model3.parameters())
train_losses = train(model=model3,
                     optimizer=optimizer,
                     dataloader=trainloader,
                     epochs=2)

triD_visualize_encoder(model3, trainloader)

plt.figure(figsize=(15, 5))
train_nums = np.array(range(len(train_losses)))
plt.plot(train_nums*100,train_losses)
plt.show()
