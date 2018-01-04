import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from random import *
from skimage import io, transform
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.mnist import MNIST
#from utils.utils import progress_bar

torch.manual_seed(8)
torch.cuda.manual_seed(8)

parser = argparse.ArgumentParser(description='model selection')
parser.add_argument('--model', choices=['linear', 'lstm', 'conv', 'bi_lstm', 'atten', 'self_atten'], default='lstm')
args = parser.parse_args()

train = torch.load('train')
test = torch.load('test')

sequence_len = 5
batch_size = 100

def index_list(data):	# get sequence index for each person and activities

	id1 = data[0,561]
	id2 = data[0,562]
	start = 0
	end = 0
	index = []

	for i in range(data.size(0)):

	  if id1 != data[i,561] or id2 != data[i,562]:
	    id1 = data[i,561]
	    id2 = data[i,562]
	    end = i - 1
	    index.append([start, end])
	    start = i

	index.append([start, data.size(0)-1])

	return index

def batch(data, index):

	inputs = torch.Tensor(sequence_len, batch_size, 561)
	label = torch.LongTensor(batch_size)

	for i in range(batch_size):
	  idx = randint(0, len(index)-1)
	  start = randint(index[idx][0], index[idx][1] - sequence_len)
	  end = start + sequence_len

	  inputs[:,i,:] = data[start:end,:561]
	  label[i] = int(data[start,562])

	return Variable(inputs.cuda()), Variable(label.cuda())

def testset(data, index):

	inputs = torch.Tensor(sequence_len, len(index), 561)
	label = torch.LongTensor(len(index))

	for i in range(len(index)):
	  start = randint(index[i][0], index[i][1] - sequence_len)
	  end = start + sequence_len

	  inputs[:,i,:] = data[start:end,:561]
	  label[i] = int(data[start,562])

	return Variable(inputs.cuda()), Variable(label.cuda())

def test_acc(input, target):

	_, pred = input.data.max(1)
	acc = pred.float().view_as(target).eq(target.data.float())

	return acc.sum()

class linear_model(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(linear_model, self).__init__()

	self.hidden_size	= hidden_size

	self.net		= nn.Sequential(
          nn.Linear(input_size, hidden_size),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_size, hidden_size),
          nn.ReLU(inplace=True),
          nn.Linear(hidden_size, hidden_size),
          nn.ReLU(inplace=True),
        )
	self.activities		= nn.Linear(hidden_size, 6)

    def forward(self, input):

	x = torch.squeeze(input)
	h = self.net(x)

	return self.activities(h)

class convolution(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(convolution, self).__init__()

	self.hidden_size	= hidden_size

	self.net		= nn.Sequential(
          nn.Conv1d(561, hidden_size, 3),
          nn.ReLU(inplace=True),
          nn.Conv1d(hidden_size, hidden_size, 3),
          nn.ReLU(inplace=True),
        )
	self.activities		= nn.Linear(hidden_size, 6)

    def forward(self, input):

	x = input.permute(1, 2, 0).contiguous()
	h = self.net(x)
	h = torch.squeeze(h)

	return self.activities(h)

class lstm(nn.Module):

    def __init__(self, input_size, hidden_size, layer):
        super(lstm, self).__init__()

	self.hidden_size	= hidden_size
	self.layer		= layer

	self.lstm		= nn.LSTM(input_size, hidden_size, layer, dropout=0.5)
	self.activities		= nn.Linear(hidden_size, 6)

    def forward(self, input):

	hidden = (Variable(torch.zeros(self.layer, input.size(1), self.hidden_size)).cuda(),
		  Variable(torch.zeros((self.layer, input.size(1), self.hidden_size))).cuda())

	h, hidden = self.lstm(input, hidden)
	h = h[-1]

	return self.activities(h)

class bidirectional_lstm(nn.Module):

    def __init__(self, input_size, hidden_size, layer):
        super(bidirectional_lstm, self).__init__()

	self.hidden_size	= hidden_size
	self.layer		= layer

	self.lstm		= nn.LSTM(input_size, hidden_size, layer, dropout=0.5, bidirectional=True) # 2 layers LSTM
	self.activities		= nn.Linear(hidden_size * sequence_len * 2, 6)

    def forward(self, input):

	hidden = (Variable(torch.zeros(self.layer * 2, input.size(1), self.hidden_size)).cuda(),
		  Variable(torch.zeros((self.layer * 2, input.size(1), self.hidden_size))).cuda())

	h, hidden = self.lstm(input, hidden)
	h = h.permute(1, 0, 2).contiguous().view(h.size(1), -1)

	return self.activities(h)

class attention(nn.Module):

    def __init__(self, input_size, hidden_size, layer):
        super(attention, self).__init__()

	self.hidden_size	= hidden_size
	self.layer		= layer

	self.token		= nn.Linear(input_size, hidden_size)
	self.lstm		= nn.LSTM(hidden_size, hidden_size, layer, dropout=0.5) # 2 layers LSTM
	self.attention		= nn.Sequential(
        				  nn.Linear(hidden_size * 2, hidden_size, bias=False),
				          nn.ReLU(inplace=True),
				          nn.Linear(hidden_size, 1, bias=False),
					  nn.Softmax(0)
			        )
	self.activities		= nn.Linear(hidden_size * sequence_len, 6)

	self.x_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))
	self.q_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))
	self.p_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))

	self.bias_1 = nn.Parameter(torch.randn(1, 1, 1))
	self.bias_2 = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, input):

	hidden = (Variable(torch.zeros(self.layer, input.size(1), self.hidden_size)).cuda(),
		  Variable(torch.zeros((self.layer, input.size(1), self.hidden_size))).cuda())

	token = self.token(input)

	h, hidden = self.lstm(token, hidden)

	# query
	q = h[:,:,None,:] * self.q_weight
	q = q.sum(-1)

	x = token[:,:,None,:] * self.x_weight
	x = x.sum(-1)

	# additive attention
	p = F.relu(q + x[:,None,:,:] + self.bias_1)
	p = p[:,:,:,None,:] * self.p_weight
	p = p.sum(-1) + self.bias_2
	p = F.softmax(p,1)

	# self attention probability
	px = p * x[None,:,:,:]
	px =px.sum(0).permute(1, 0, 2).contiguous().view(h.size(1), -1)

	return self.activities(px)

class self_attention(nn.Module):

    def __init__(self, input_size, hidden_size, layer):
        super(self_attention, self).__init__()

	self.hidden_size	= hidden_size
	self.layer		= layer

	self.token		= nn.Linear(input_size, hidden_size)
	self.attention		= nn.Sequential(
        				  nn.Linear(hidden_size * 2, hidden_size, bias=False),
				          nn.ReLU(inplace=True),
				          nn.Linear(hidden_size, 1, bias=False),
					  nn.Softmax(0)
			        )
	self.activities		= nn.Linear(hidden_size * sequence_len, 6)

	self.x_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))
	self.q_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))
	self.p_weight = nn.Parameter(torch.randn(1, 1, hidden_size, hidden_size))

	self.bias_1 = nn.Parameter(torch.randn(1, 1, 1))
	self.bias_2 = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, input):

	hidden = (Variable(torch.zeros(self.layer, input.size(1), self.hidden_size)).cuda(),
		  Variable(torch.zeros((self.layer, input.size(1), self.hidden_size))).cuda())

	token = self.token(input)

	# query
	q = token[:,:,None,:] * self.q_weight
	q = q.sum(-1)

	x = token[:,:,None,:] * self.x_weight
	x = x.sum(-1)

	# additive attention
	p = F.relu(q + x[:,None,:,:] + self.bias_1)
	p = p[:,:,:,None,:] * self.p_weight
	p = p.sum(-1) + self.bias_2
	p = F.softmax(p,1)

	# self attention probability
	px = p * x[None,:,:,:]
	px =px.sum(0).permute(1, 0, 2).contiguous().view(token.size(1), -1)

	return self.activities(px)

class criterion(nn.Module):
    def __init__(self):
        super(criterion, self).__init__()

        self.activities_loss = nn.CrossEntropyLoss()

    def forward(self, activities, activities_target):

	return self.activities_loss(activities, activities_target)

if args.model == 'linear':
	model = linear_model(561, 500).cuda()
	sequence_len = 1
elif args.model == 'conv':
	model = convolution(561, 50).cuda()
elif args.model == 'lstm':
	model = lstm(561, 50, 2).cuda()
elif args.model == 'bi_lstm':
	model = bidirectional_lstm(561, 50, 2).cuda()
elif args.model == 'atten':
	model = attention(561, 50, 2).cuda()
elif args.model == 'self_atten':
	model = self_attention(561, 50, 2).cuda()

print 'running model', args.model

optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss = criterion()

train_index = index_list(train)
test_index = index_list(test)

for epoch in xrange(30000):

	model.train()
	optimizer.zero_grad()

	inputs, label_activities = batch(train, train_index)

	pred = model(inputs)

	train_loss = loss(pred, label_activities)
	train_loss.backward()
	torch.nn.utils.clip_grad_norm(model.parameters(), 10)

	optimizer.step()

	if epoch % 100 == 0:

	  model.eval()
	  acc_train = test_acc(pred, label_activities)

	  test_inputs, test_label_activities = testset(test, test_index)

	  test_pred = model(test_inputs)

	  acc_test = test_acc(test_pred, test_label_activities)

	  print 'epoch', epoch
	  print 'training acc', "%6.3f"% (acc_train/100.), 'training loss', train_loss.data[0]
	  print 'test acc', "%10.3f"% (acc_test/120.)


