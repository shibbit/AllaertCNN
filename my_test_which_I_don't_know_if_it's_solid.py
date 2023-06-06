from natsort import natsorted, ns
import torchvision.transforms as transforms
import torch
import allCNN
import os
import numpy as np
from torch.autograd import Variable
import get_folder_and_file as get
from skimage import io
from PIL import Image
import utils
import random

seeds = 1
torch.cuda.manual_seed_all(seeds)
torch.manual_seed(seeds)
np.random.seed(seeds)
random.seed(seeds)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(seeds)

transform_test = transforms.Compose([
	transforms.Resize(96),
	# transforms.RandomCrop(80),
	# transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
	transforms.TenCrop(80),
	transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
])  # 不懂

use_cuda = True


def load_model(model_Path, model_Name):
	net = allCNN.allCNN()
	checkpoint = torch.load(os.path.join(model_Path, model_Name))
	net.load_state_dict(checkpoint['net'])
	net.cuda()
	net.eval()
	return net


def test(test_Path, model):
	global total_accuracy, images
	correct = 0
	images = []
	labels = []
	label_num_list = []
	"""get all images and labels in the designated path"""
	print("\nReading images from %s" % test_Path)
	folders = get.get_folders(test_Path)
	folders.sort()
	for index, folder in enumerate(folders):
		list_ = os.listdir(folder)
		files = natsorted(list_, alg=ns.PATH)
		num_of_current_label = len(files)
		label_num_list.append(num_of_current_label)
		for i in files:
			images.append(os.path.join(folder, i))
			label = os.path.split(folder.replace(os.path.dirname(folder), ''))[1]
			labels.append(label)
	
	"""convert labels to value"""
	for j in range(len(label_num_list)):
		if j == 0:
			stop = label_num_list[0]
			for k in range(stop):
				labels[k] = 0
		else:
			start = sum(label_num_list[0:j])
			stop = sum(label_num_list[0:j + 1])
			for l in range(start, stop):
				labels[l] = j
	
	"""iterate all images and predict"""
	model = model
	print("predicting ...")
	for l in range(len(images)):
	# for l in range(15000):
		raw_img = io.imread(images[l])
		img = Image.fromarray(raw_img)
		inputs = transform_test(img)
		ncrops, c, h, w = np.shape(inputs)
		# c, h, w = np.shape(inputs)
		inputs = inputs.view(-1, c, h, w)
		# inputs = inputs.view(c, h, w)
		inputs = inputs.cuda()
		with torch.no_grad():
			inputs = Variable(inputs)
		outputs = model(inputs)
		outputs_avg = outputs.view(ncrops, -1).mean(0)
		_, predicted = torch.max(outputs_avg.data, 0)
		# predicted = torch.max(outputs, 0)
		if predicted == labels[l]:
			correct += 1
		else:
			continue
		utils.progress_bar(l, len(images), None, 'Progress:')
	accuracy = correct / len(images)
	print("\nAccuracy: %.5f" % accuracy)


if __name__ == '__main__':
	datapath = r'C:\Users\guote\PycharmProjects\AllaertCNN\vqvae2reconstructed120'
	# datapath = r'C:\Users\guote\PycharmProjects\AllaertCNN\categorized-m'
	trained = load_model(r'C:\Users\guote\PycharmProjects\AllaertCNN\modelsave', 'Test_model_5-14_backup.t7')
	test(datapath, trained)
