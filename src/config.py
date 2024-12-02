import torch
import os
from torch import cuda
from src.utils import parse_code
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# from increment_refdata_use import get_data


class config:
    def __init__(self):
        self.model_path = '/home/danfossautonomypc/segmentation/checkpoints/'  # todo: replace with registry later
        
        self.batch = 32
        self.lr = 0.001
        self.epochs = 18
        self.input_size = (128, 128)
        

	
# class config:
	
	
# 	# def __init__(self):
	
# 		# artifact = self.get_artifact
# 	# train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list = get_data()
	

# 	model_path = '/home/danfossautonomypc/segmentation/checkpoints/' #todo: replace with registry later
# 	load_model = model_path+'state_dict18s.pt'
# 	# model_path = "./Unet/model/"
# 	# path = "./Unet/Dataset/"
# 	# load_model = "./Unet/model/state_dict.pt"
# 	batch = 32
# 	lr = 0.002
# 	epochs = 12
# 	input_size = (128,128)
# 	if cuda.is_available(): device = torch.device("cuda")
# 	else: device = torch.device('cpu')
# 	# code2id, id2code, name2id, id2name = Color_map(path+'class_dict.csv')
	