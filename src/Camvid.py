
import natsort
import os
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from src.utils import *
from src.config import config

# CONFIG = config()
# id2code = CONFIG.id2code


class CamSeqDataset(Dataset):
	def __init__(self, image_dir, mask_dir, images, masks, image_size,id2code):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.images = images
		# self.images = [img for img in os.listdir(image_dir) if img.endswith('.png')]

		# self.masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in self.images]
		self.masks = masks
		self.total_imgs = natsort.natsorted(self.images)
		self.total_masks = natsort.natsorted(self.masks)
		self.image_size = image_size

		self.image_transform = transforms.Compose([transforms.Resize(image_size, 0)])
		self.id2code =id2code


	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_name = self.total_imgs[idx]
		img_path = os.path.join(self.image_dir, img_name)
		

		image = Image.open(img_path).convert('RGB')
		
		image = self.image_transform(image)
		out_image = transforms.Compose([transforms.ToTensor()])(image) 

		mask_name = self.total_masks[idx]
		mask_path = os.path.join(self.mask_dir, mask_name)
		

		mask = Image.open(mask_path).convert('RGB')
		
		mask = self.image_transform(mask)
		rgb_mask = transforms.Compose([transforms.PILToTensor()])(mask)
		# print("sssssssssssssss",rgb_mask.shape)


		out_mask = rgb_to_onehot(torch.from_numpy(np.array(rgb_mask)).permute(1,2,0), self.id2code)


		return out_image, out_mask, rgb_mask.permute(0,1,2)


class Test():
	def __init__(self, image_dir, mask_dir, image_size):
		self.image_dir = image_dir
		self.mask_dir = mask_dir
		self.images = [img for img in os.listdir(image_dir) if img.endswith('.png')]

		self.masks = [img_name[:-4] + '_L' + img_name[-4:] for img_name in self.images]
		self.total_imgs = natsort.natsorted(self.images)
		self.total_masks = natsort.natsorted(self.masks)
		self.image_size = image_size

		self.image_transform = transforms.Compose([transforms.Resize(image_size, 0)])


	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		img_name = self.total_imgs[idx]
		img_path = os.path.join(self.image_dir, img_name)
		

		image = Image.open(img_path).convert('RGB')
		
		image = self.image_transform(image)
		out_image = transforms.Compose([transforms.ToTensor()])(image) 

		mask_name = self.total_masks[idx]
		mask_path = os.path.join(self.mask_dir, mask_name)
		

		mask = Image.open(mask_path).convert('RGB')
		
		rgb_mask = self.image_transform(mask)
		rgb_mask = transforms.Compose([transforms.PILToTensor()])(rgb_mask)
		# print("sssssssssssssss",rgb_mask.shape)


		# out_mask = rgb_to_onehot(torch.from_numpy(np.array(rgb_mask)).permute(1,2,0), id2code)


		return out_image, rgb_mask

