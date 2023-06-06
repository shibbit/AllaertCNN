import torch.nn as nn
import torch.nn.functional as F
# cfg = {
#     'AllaertCNN': ['M', 16, 'M', 32, 'M'],
# 		}

class allCNN(nn.Module):
	def __init__(self):
		super(allCNN,self).__init__()
		self.features = self._make_layers()
		self.classfier = nn.Linear(3200,7)
		# self.classfier = nn.Linear(800, 7)
	def forward(self,x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = self.classfier(out)
		return out
		
	def _make_layers(self):
		layers = []
		layers = [nn.Conv2d(3,8,kernel_size=5,stride=1,padding=2),
				  nn.BatchNorm2d(8),
				  nn.ReLU(inplace=True),
				  nn.MaxPool2d(kernel_size=2, stride=2),
				  nn.Conv2d(8,16,kernel_size=3,stride=1,padding=1),
				  nn.BatchNorm2d(16),
				  nn.ReLU(inplace=True),
				  nn.MaxPool2d(kernel_size=2, stride=2),
				  nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
				  nn.BatchNorm2d(32),
				  nn.ReLU(inplace=True),
				  nn.MaxPool2d(kernel_size=2, stride=2)
				  ]
		
		return nn.Sequential(*layers)