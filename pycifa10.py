import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from visdom import Visdom
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.cifar import CIFAR10
from libs.utils import progress_bar

torch.manual_seed(8)
torch.cuda.manual_seed(8)

class capsnet(nn.Module):
# capsnet( # primary capsule, # routing, # conv1, # plane per capsule , linear capsule dim, # class )
    def __init__(self, nCapsule, nIte, nInputPlane, nOutputPlane, outputDimesion, nOutput):
        super(capsnet, self).__init__()

	self.nCapsule = nCapsule
	self.nIte = nIte
	self.nInputPlane = nInputPlane
	self.nOutputPlane = nOutputPlane
	self.outputDimesion = outputDimesion
	self.nOutput = nOutput

        self.conv1 = nn.Sequential(
          nn.Conv2d(3,nInputPlane,9),
          nn.ReLU(inplace=True),
	  nn.Conv2d(nInputPlane, nCapsule * nOutputPlane, 8, 2)
        )

	self.primary_capsules = nn.ModuleList(
	  [nn.Conv2d(nInputPlane, nOutputPlane, 8, 2)
           for _ in range(nCapsule)
          ]
        )

	self.route_weight = nn.Parameter(torch.randn(nCapsule, nOutput, outputDimesion, nOutputPlane, 25))

	self.decoder = nn.Sequential(
          nn.Linear(outputDimesion * nOutput, 512),
          nn.ReLU(inplace=True),
          nn.Linear(512, 1024),
          nn.ReLU(inplace=True),
          nn.Linear(1024, 3200),
          nn.ReLU(inplace=True),
        )

	self.decon = nn.Sequential(
          nn.ConvTranspose2d(128,256,8,2),
          nn.ReLU(inplace=True),
          nn.ConvTranspose2d(256,3,9),
          nn.Sigmoid(),
        )

    def onehot(self, input):
	return torch.eye(self.nOutput).cuda().index_select(dim=0, index=input.long())

    def squash(self, input):
	scale = torch.norm(input, 2, -1) / (1 + torch.norm(input, 2, -1)) / torch.norm(input, 1, -1)
	return scale[:,:,None] * input

    def route(self, input):
	b = Variable(torch.zeros(input.size(0), self.nCapsule, self.nOutput, 1, 25)).cuda()

	for i in range(self.nIte):
          b = F.softmax(b,1)
	  v = input * b
	  v = v.sum(-1).sum(-3)
	  v = self.squash(v)
	  db = input * v[:,None,:,:,None]
	  b = b + db.sum(-2,keepdim=True)

	return v

    def forward(self, input, target):
        u = self.conv1(input)
	u = self.squash(u.view(u.size(0), self.nCapsule, -1))
	u = u.view(u.size(0), self.nCapsule, 1, 1, self.nOutputPlane, 25)
	u_hat = u * self.route_weight
	u_hat = u_hat.sum(-2)

	v = self.route(u_hat)

	onehot = Variable(self.onehot(target))
	output = v * onehot[:,:,None]
	output = self.decoder(output.view(output.size(0),-1))
	output = self.decon(output.view(output.size(0),128,5,5))

        return v, output, onehot

class criterion(nn.Module):
    def __init__(self):
        super(criterion, self).__init__()

        self.decoder_loss = nn.MSELoss()

    def forward(self, input, pred, target, reconst=None):
	mseloss = self.decoder_loss(reconst, input)

	left = F.relu(0.9 - torch.norm(pred, 1, -1), inplace=True) ** 2
	right = F.relu(torch.norm(pred, 1, -1) - 0.1, inplace=True) ** 2
	marginloss = target * left + 0.5 * (1 - target) * right
	loss = 0.0005 * mseloss + marginloss.sum() / input.size(0)

	return loss

class predition(nn.Module):
    def __init__(self):
        super(predition, self).__init__()

    def forward(self, pred, target, cm=None):
	output = torch.norm(pred, 2, -1)
	_, output = output.data.max(1, keepdim=True)
	acc = output.float().view_as(target).eq(target.float())
	return acc.sum()

image_transform = transforms.Compose([
    transforms.RandomCrop(24),
    transforms.ToTensor(),
])

cifar_train = CIFAR10(root='datasets/cifar10', download=True, train=True, transform=image_transform)
cifar_valid = CIFAR10(root='datasets/cifar10', download=True, train=False, transform=image_transform)

train = torch.utils.data.DataLoader(dataset=cifar_train, batch_size=128, shuffle=True, num_workers=2)
valid = torch.utils.data.DataLoader(dataset=cifar_valid, batch_size=100, shuffle=False, num_workers=2)

# capsnet( # primary capsule, # routing, # conv1, # plane per capsule , linear capsule dim, # class )
model = capsnet(64, 3, 512, 8, 16, 10)
marginloss = criterion()
optimizer = optim.Adam(model.parameters(), lr=0.01)
match = predition()


for epoch in xrange(30):
	print "epoch", epoch + 1
	train_acc = 0.
	model.train()
	for batch_idx, (x, y) in enumerate(train):
	  optimizer.zero_grad()

	  input, target = Variable(x).cuda(), y.cuda()

	  v, reconst, onehot = model(input, target)
	  loss = marginloss(input, v, onehot, reconst)
	  loss.backward()

	  train_acc = match(v, target)

	  optimizer.step()

	  progress_bar(batch_idx, len(train), 't loss: %.3f | acc: %.3f%%' % (loss, train_acc/float(y.size(0))))

	valid_loss, valid_acc = 0., 0.
	model.eval()
	for batch_idx, (x, y) in enumerate(valid):
	  input, target = Variable(x).cuda(), y.cuda()

	  v, reconst, onehot = model(input, target)
	  valid_loss += marginloss(input, v, onehot, reconst).data

	  valid_acc += match(v, target)

	  progress_bar(batch_idx, len(valid), 'v loss: %.3f | acc: %.3f%%' % (valid_loss/(batch_idx+1), valid_acc/(batch_idx+1)/100.))

	  ori[batch_idx] = x[0]
	  res[batch_idx] = reconst[0].data


