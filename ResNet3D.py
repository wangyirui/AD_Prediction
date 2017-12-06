import torch
import torch.nn as nn
import torch.nn.functional as F

class Sideway(nn.Module):

	def __init__(self, features):
		super(Sideway, self).__init__()
		self.bn = nn.BatchNorm3d(num_features = features)
		self.conv = nn.Conv3d(  in_channels = features,
								out_channels = features,
								kernel_size = 3,
								stride = 1,
								padding = 1)
	def forward(self, out):
		out = F.relu(self.bn(out))
		out = F.relu(self.bn(self.conv(out)))
		out = self.conv(out)
		return out

class ResNet(nn.Module):

	def __init__(self):
		super(ResNet, self).__init__()
		self.conv1_0 = nn.Conv3d(in_channels = 1,
								out_channels = 32,
								kernel_size = 3,
								stride = 1,
								padding = 1)
		self.conv1_1 = nn.Conv3d( in_channels = 32,
								out_channels = 32,
								kernel_size = 3,
								stride = 1,
								padding = 1)
		self.bn1_0 = nn.BatchNorm3d(num_features = 32)
		self.bn1_1 = nn.BatchNorm3d(num_features = 32)
		self.conv2_0 = nn.Conv3d(in_channels = 32,
								out_channels = 64,
								kernel_size = 3,
								stride = 2,
								padding = 1)
		self.conv2_1 = nn.Conv3d(in_channels = 64,
								out_channels = 64,
								kernel_size = 3,
								stride = 2,
								padding = 1)
		self.sideway1_0 = Sideway(features = 64)
		self.sideway1_1 = Sideway(features = 64)
		self.sideway1_2 = Sideway(features = 64)
		self.sideway1_3 = Sideway(features = 64)
		self.bn2_0 = nn.BatchNorm3d(num_features = 64)
		self.bn2_1 = nn.BatchNorm3d(num_features = 64)
		self.conv3 = nn.Conv3d( in_channels =64,
								out_channels = 128,
								kernel_size = 3,
								stride = 2,
								padding =1)
		self.sideway2_0 = Sideway(features = 128)
		self.sideway2_1 = Sideway(features = 128)
		self.pool = nn.MaxPool3d(kernel_size = 7,
								stride = 1)
		self.fc1 = nn.Linear(in_features = 128, 
							 out_features = 128)
		self.fc2 = nn.Linear(in_features = 128,
							 out_features = 3)
		self.softmax = nn.Softmax()


	def forward(self, out):
		out = F.relu(self.bn1_0(self.conv1_0(out)))
		out = F.relu(self.bn1_1(self.conv1_1(out)))
		out = self.conv2_0(out)
		out_s = self.sideway1_0(out)

		out_s = self.sideway1_1(out+out_s)

		out = F.relu(self.bn2_0(out+out_s))
		out = self.conv2_1(out)
		out_s = self.sideway1_2(out)

		out_s = self.sideway1_3(out+out_s)

		out = F.relu(self.bn2_1(out+out_s))
		out = self.conv3(out)
		out_s = self.sideway2_0(out)

		out_s = self.sideway2_1(out+out_s)

		out_ = self.pool(out+out_s)
		out = out_.view(out_.size(0), 128)
		out = F.relu(self.fc1(out))
		out = self.softmax(self.fc2(out))
		return out
