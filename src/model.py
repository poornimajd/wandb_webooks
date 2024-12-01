import torch
import torch.nn as nn
import torch.nn.functional as F




class UNet(nn.Module):
	def __init__(self,n_filters = 32, bn = True, dilation_rate = 1):
		super(UNet, self).__init__()
		self.bn = bn

		#downsampling
		self.conv1_1 =nn.Conv2d(3, n_filters, kernel_size = 3, padding=1, dilation  = dilation_rate)
		self.conv1_2 = nn.Conv2d(n_filters, n_filters, kernel_size = 3, padding=1, dilation = dilation_rate)
		self.bn1 = nn.BatchNorm2d(n_filters) if bn else None

		self.conv2_1 = nn.Conv2d(n_filters, n_filters*2, kernel_size = 3, padding = 1, dilation = dilation_rate)
		self.conv2_2 = nn.Conv2d(n_filters*2, n_filters*2, kernel_size = 3, padding =1, dilation = dilation_rate)
		self.bn2 = nn.BatchNorm2d(n_filters*2) if bn else None

		self.conv3_1 = nn.Conv2d(n_filters*2, n_filters*4, kernel_size = 3, padding = 1, dilation = dilation_rate)
		self.conv3_2 = nn.Conv2d(n_filters*4, n_filters*4, kernel_size = 3, padding =1, dilation = dilation_rate)
		self.bn3 = nn.BatchNorm2d(n_filters*4) if bn else None

		self.conv4_1 = nn.Conv2d(n_filters*4, n_filters*8, kernel_size = 3, padding = 1, dilation = dilation_rate)
		self.conv4_2 = nn.Conv2d(n_filters*8, n_filters*8, kernel_size = 3, padding =1, dilation = dilation_rate)
		self.bn4 = nn.BatchNorm2d(n_filters*8) if bn else None

		self.conv5_1 = nn.Conv2d(n_filters*8, n_filters*16, kernel_size = 3, padding = 1, dilation = dilation_rate)
		self.conv5_2 = nn.Conv2d(n_filters*16, n_filters*16, kernel_size = 3, padding =1, dilation = dilation_rate)
		self.bn5 = nn.BatchNorm2d(n_filters*16) if bn else None

		#upsampling
		
		self.conv6_1 =nn.Conv2d(n_filters*24, n_filters*8,kernel_size=3,padding=1, dilation=dilation_rate)
		self.conv6_2 = nn.Conv2d(n_filters*8, n_filters*8, kernel_size =3, padding=1, dilation=dilation_rate)
		self.bn6 = nn.BatchNorm2d(n_filters*8) if bn else None

		
		self.conv7_1 = nn.Conv2d(n_filters*12, n_filters*4, kernel_size=3, padding=1, dilation=dilation_rate)
		self.conv7_2 =nn.Conv2d(n_filters*4, n_filters*4, kernel_size=3, padding=1, dilation=dilation_rate)
		self.bn7 = nn.BatchNorm2d(n_filters*4) if bn else None

		
		self.conv8_1 = nn.Conv2d(n_filters*6, n_filters*2, kernel_size=3, padding=1, dilation=dilation_rate)
		self.conv8_2 = nn.Conv2d(n_filters*2, n_filters*2, kernel_size=3, padding=1, dilation=dilation_rate)
		self.bn8 = nn.BatchNorm2d(n_filters*2) if bn else None

        
		self.conv9_1 = nn.Conv2d(n_filters*3, n_filters, kernel_size=3, padding=1, dilation=dilation_rate)
		self.conv9_2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, dilation=dilation_rate)
		self.bn9 = nn.BatchNorm2d(n_filters) if bn else None

		self.final_conv = nn.Conv2d(n_filters, 32, kernel_size=1)
	
	def forward(self,x):
		# print("sssssssssssss",x.shape)
		conv1 = F.relu(self.conv1_1(x))
		if self.bn: conv1 = self.bn1(conv1)
		conv1 =F.relu(self.conv1_2(conv1))
		if self.bn: conv1 = self.bn1(conv1)
		pool1 = F.max_pool2d(conv1, 2)

		conv2 = F.relu(self.conv2_1(pool1))
		if self.bn: conv2 = self.bn2(conv2)
		conv2 = F.relu(self.conv2_2(conv2))
		if self.bn: conv2 =self.bn2(conv2)
		pool2 = F.max_pool2d(conv2, 2)

		conv3 = F.relu(self.conv3_1(pool2))
		if self.bn: conv3 = self.bn3(conv3)
		conv3 = F.relu(self.conv3_2(conv3))
		if self.bn: conv3 =self.bn3(conv3)
		pool3 = F.max_pool2d(conv3, 2)


		conv4 = F.relu(self.conv4_1(pool3))
		if self.bn: conv4 = self.bn4(conv4)
		conv4 = F.relu(self.conv4_2(conv4))
		if self.bn: conv4 =self.bn4(conv4)
		pool4 = F.max_pool2d(conv4, 2)


		conv5 = F.relu(self.conv5_1(pool4))
		if self.bn: conv5 = self.bn5(conv5)
		conv5 = F.relu(self.conv5_2(conv5))
		if self.bn: conv5 = self.bn5(conv5)

		#upsamplin

		up6 = torch.cat([F.interpolate(conv5, scale_factor = 2, mode ='bilinear',align_corners=True), conv4], dim=1)
		conv6 = F.relu(self.conv6_1(up6))
		if self.bn: conv6 = self.bn6(conv6)
		conv6 = F.relu(self.conv6_2(conv6))
		if self.bn: conv6 = self.bn6(conv6)


		up7 = torch.cat([F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=True), conv3], dim=1)
		conv7 = F.relu(self.conv7_1(up7))
		if self.bn: conv7 = self.bn7(conv7)
		conv7 = F.relu(self.conv7_2(conv7))
		if self.bn: conv7 = self.bn7(conv7)


		up8 = torch.cat([F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True), conv2], dim=1)
		conv8 = F.relu(self.conv8_1(up8))
		if self.bn: conv8 = self.bn8(conv8)
		conv8 = F.relu(self.conv8_2(conv8))
		if self.bn: conv8 = self.bn8(conv8)

		up9 = torch.cat([F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=True), conv1], dim=1)
		conv9 = F.relu(self.conv9_1(up9))
		if self.bn: conv9 = self.bn9(conv9)
		conv9 = F.relu(self.conv9_2(conv9))
		if self.bn: conv9 = self.bn9(conv9)


		output = self.final_conv(conv9)  # Return raw logits

		return output

