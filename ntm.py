import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skimage import io, transform
from torch.autograd import Variable

torch.manual_seed(8)
torch.cuda.manual_seed(8)

def sequence(): # priority sort sequence and target generator
	d = np.random.permutation(256)
	d = d[:20]

	p = torch.Tensor(20,1).uniform_(-1, 1)
	s = torch.from_numpy((((d[:,None] & (1 << np.arange(8)))) > 0).astype(int)).float()
	input_padding = torch.zeros(16,9)

	input = Variable(torch.cat( (torch.cat((p, s),1), input_padding), 0 )[:,None,:])

	_, idx = torch.sort(p, 0, True)
	target_padding = torch.zeros(20,8)

	target = Variable(torch.cat( (target_padding, s.index_select(0,torch.squeeze(idx))), 0)[:36,None,:])

	return input.cuda(), target.cuda()

def batch():
	batch_input = []
	batch_target = []
	for i in range(200):
	  x, y = sequence()
	  batch_input.append(x)
	  batch_target.append(y)

	return torch.cat(batch_input, 1), torch.cat(batch_target, 1)

class memory_bank(nn.Module):

    def __init__(self, n_read, n_write, input_size, m_dim, kernel_size):
        super(memory_bank, self).__init__()

	self.n_read		= n_read
	self.n_write		= n_write
	self.m_dim		= m_dim
	self.kernel_size	= kernel_size
	self.kernel_shift	= np.floor(self.kernel_size / 2).astype(int)

	self.read_head		= nn.Linear(input_size, (m_dim + 3 + kernel_size) * n_read)
	self.write_head		= nn.Linear(input_size, (m_dim + 3 + kernel_size) * n_write)
	self.erase_add		= nn.Linear(input_size, 2 * m_dim * n_write)

    def address(self, input, mem, old_key, n_head):
	key = F.relu(input[:,:,:self.m_dim])
	beta = F.softplus(input[:,:,self.m_dim])
	gate = F.sigmoid(input[:,:,-5])
	gamma = F.relu(input[:,:,-4]) + 1
	shift = F.softmax(input[:,:,-3:],2)

	# content addressing
	dot = mem[:,None,:,:] * key[:,:,None,:]
	wc = torch.norm(mem, 2, -1)[:,None,:] * torch.norm(key, 2, -1, keepdim=True) + 1e-12
	wc = beta[:,:,None] * dot.sum(-1) / wc
	wc = F.softmax(wc,2)

	# interpolation
	wg = wc * gate[:,:,None] + (1 - gate[:,:,None]) * old_key

	# convolution shift
	w = torch.cat((wg[:,:,-self.kernel_shift:], wg, wg[:,:,:self.kernel_shift]),2)
	wt = F.conv1d(w.view(1,n_head * w.size(0), w.size(2)), shift.view(n_head * w.size(0), 1, 3), groups=n_head * w.size(0))

	# sharpening
	wt = wt.view(w.size(0),n_head, wt.size(2)) ** gamma[:,:,None]
	new_w = wt / wt.sum(-1, keepdim=True)

	return new_w

    def forward(self, input, mem, old_read_key, old_write_key):
	wk = self.read_head(input).view(input.size(0), self.n_read, -1)
	ww = self.address(wk, mem, old_write_key, self.n_write)

	v = self.erase_add(input).view(input.size(0), self.n_write, 2, -1)
	erase_vector = F.sigmoid(v[:,:,0,:])
	add_vector = F.relu(v[:,:,1,:])


	we = 1 - ww[:,:,:,None] * erase_vector[:,:,None,:]

	wa = ww[:,:,:,None] * add_vector[:,:,None,:]

	new_mem = torch.cumprod(we, 1)[:,-1,:,:] + wa.sum(1)

	rk = self.read_head(input).view(input.size(0), self.n_read, -1)
	rw = self.address(rk, new_mem, old_read_key, self.n_read)

	output = new_mem[:,None,:,:] * rw[:,:,:,None]
	output = output.sum(-2)

	return output, new_mem, rw, ww

class ntm(nn.Module):

    def __init__(self, input_size, hidden_size, n_read, n_write, n_dim, m_dim, kernel_size):
        super(ntm, self).__init__()

	self.hidden_size	= hidden_size
	self.n_read		= n_read
	self.n_write		= n_write
	self.n_dim		= n_dim
	self.m_dim		= m_dim
	self.lstm_input_size	= 2 * (input_size + n_read * m_dim)

	self.memory_bank	= memory_bank(n_read, n_write, hidden_size, m_dim, kernel_size)

	self.controller		= nn.LSTM(input_size + n_read * m_dim, hidden_size, 2) # 2 layers LSTM

	self.output		= nn.Sequential(
				    nn.Linear(hidden_size, 8),
				    nn.Sigmoid(),
			        )

    def forward(self, input):
	memory = Variable(torch.zeros(input.size(1), self.n_dim, self.m_dim)).cuda()
	read_key = Variable(torch.zeros(input.size(1), self.n_read, self.n_dim)).cuda()
	write_key = Variable(torch.zeros(input.size(1), self.n_write, self.n_dim)).cuda()
	readout = Variable(torch.zeros(input.size(1), self.n_read, self.m_dim)).cuda()

	hidden = (Variable(torch.zeros(2, input.size(1), self.hidden_size)).cuda(),
		  Variable(torch.zeros((2, input.size(1), self.hidden_size))).cuda())

	# heads state initialization
	read_key[:,0,0] = 1
	read_key[:,1,1] = 1
	read_key[:,2,2] = 1
	read_key[:,3,3] = 1
	read_key[:,4,4] = 1
	write_key[:,0,0] = 1
	write_key[:,1,1] = 1
	write_key[:,2,2] = 1
	write_key[:,3,3] = 1
	write_key[:,4,4] = 1
	memory[:,0,0] = 1

	output = []

	for i in range(input.size(0)):
	  lstm_input = torch.cat((input[i], readout.view(input.size(1),-1)), 1)
	  out, hidden = self.controller(lstm_input[None,:,:], hidden)
	  readout, memory, read_key, write_key = self.memory_bank(out[0], memory, read_key, write_key)

	  output.append(self.output(out))

	return torch.cat(output,0)

criterion = nn.BCELoss()

# neural turing machine initialization parameters
# sequency size, hidden_size, n read head, n write head, memory lot size, memory vector size, shift kernel size
model = ntm(9, 100, 5, 5, 128, 20, 3).cuda()
optimizer = optim.RMSprop(model.parameters(), lr=3e-5, momentum=0.9)

for epoch in xrange(3000000):
	optimizer.zero_grad()
#	input, target = sequence() #single sample traning, extremely slow
	input, target = batch()

	pred = model(input)
	loss = criterion(pred, target)
	loss.backward()
	torch.nn.utils.clip_grad_norm(model.parameters(), 10)

	optimizer.step()

	if epoch % 1000 == 0:
	  print(target[-16:,0,:])
	  print(torch.floor(pred[-16:,0,:]+0.5))

	if epoch % 100 == 0:
	  print 'epoch', epoch + 1, 'loss:', loss.data[0] 


