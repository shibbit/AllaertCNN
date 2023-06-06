import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import random
import allCNN

import torch.optim as optim
import utils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader as loader
from torch.autograd import Variable
import torchvision.transforms as transforms
import time
modelpath = r'./modelsave'
datapath = r'C:\Users\guote\PycharmProjects\AllaertCNN\CASME2'
# datapath = r'C:\Users\guote\PycharmProjects\AllaertCNN\data'
dataset = 'CASME2'
use_cuda = torch.cuda.is_available()


transform_train = transforms.Compose([
	transforms.Resize(96),
	transforms.RandomCrop(80),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
])

transform_test = transforms.Compose([
	transforms.Resize(96),
	transforms.TenCrop(80),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])) ]) # 不懂

best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 50 # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9
lr = 0.001
seeds = 1
total_epoch = 120
torch.cuda.manual_seed_all(seeds)
torch.manual_seed(seeds)
np.random.seed(seeds)
random.seed(seeds)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seeds)

# net = allCNN.allCNN()
# net = vgg.VGG('VGG19')
net = vgg.VGG('VGG11')
if use_cuda:
	net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

trainset = ImageFolder(os.path.join(datapath,'train'),transform=transform_train)
testset = ImageFolder(os.path.join(datapath,'test'),transform=transform_test)
trainloader = loader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = loader(testset, batch_size=16, shuffle=True, num_workers=0)

def train(epoch):
	print('\nEpoch: %d' % epoch)
	global Train_acc
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	
	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = lr * decay_factor
		utils.set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = lr
	print('learning_rate: %s' % str(current_lr))
	
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		utils.clip_gradient(optimizer, 0.1)
		optimizer.step()
		
		train_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		
		utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
		
	
	Train_acc = 100. * correct / total
	

def test(epoch):
	global Test_acc
	global best_Test_acc
	global best_Test_acc_epoch
	net.eval()
	PrivateTest_loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(testloader):
		bs, ncrops, c, h, w = np.shape(inputs)
		inputs = inputs.view(-1, c, h, w)
		
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs, volatile=True), Variable(targets)
		outputs = net(inputs)
		outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
		
		loss = criterion(outputs_avg, targets)
		PrivateTest_loss += loss.item()
		x = outputs_avg
		_, predicted = torch.max(outputs_avg.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		print("Accuracy of this batch: %.3f"%(correct/total))
		utils.progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
	# Save checkpoint.
	Test_acc = 100. * correct / total
	
	
	if Test_acc > best_Test_acc:
		print('Saving..')
		print("best_Test_acc: %0.3f" % Test_acc)
		state = {'net': net.state_dict() if use_cuda else net,
				 'best_Test_acc': Test_acc,
				 'best_Test_acc_epoch': epoch,
				 }
		if not os.path.isdir(dataset + '_' + 'cnn'):
			os.mkdir(dataset + '_' + 'cnn')
		if not os.path.isdir(modelpath):
			os.mkdir(modelpath)
		torch.save(state, os.path.join(modelpath, 'Test_model.t7'))
		best_Test_acc = Test_acc
		best_Test_acc_epoch = epoch


for epoch in range(start_epoch, total_epoch):
	starter = time.perf_counter()
	train(epoch)
	test(epoch)
	end = time.perf_counter()
	print("This epoch took %d seconds to finish."%(end - starter))
print("best_Train_acc: %0.3f" % Train_acc)
print("best_Test_acc: %0.3f" % best_Test_acc)
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)