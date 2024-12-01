import wandb
# import cv2
import requests
# import wandb
import numpy as np
from PIL import Image
import io
import cv2


####################################################### if u want to add any files
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="uploading_mount_data", id="mount_upload", resume = "allow")

artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/segment_dataset_reference:v0', type='dataset')
# wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

draft_artifact = artifact.new_draft()  # create a draft version

# modify a subset of files in the draft version
draft_artifact.add_file("/home/danfossautonomypc/segmentation/readme_upload.txt")
# draft_artifact.remove("dir_to_remove/")
run.log_artifact(
    draft_artifact
) 

################################################################### else just use the below snippet

# def get_data():
#     train_images_list = []
#     val_images_list=[]
#     train_masks_list=[]
#     val_masks_list=[]
#     val_masks_dir = None
#     train_masks_dir = None
#     val_images_dir = None
#     train_images_dir =None
#     for k,v in artifact.manifest.entries.items():
#     #   print(v.ref
#         if v.ref is not None:
#             if 'val_masks' in v.ref:
#                 val_masks_dir = '/'.join(v.ref[7:].split('/')[:-1])
#                 val_masks_list.append(v.ref.split('/')[-1])
#             if 'train_masks' in v.ref:
#                 train_masks_dir = '/'.join(v.ref[7:].split('/')[:-1])
#                 train_masks_list.append(v.ref.split('/')[-1])
#             if 'val_frames' in v.ref:
#                 val_images_dir = '/'.join(v.ref[7:].split('/')[:-1])
#                 val_images_list.append(v.ref.split('/')[-1])
#             if 'train_masks' in v.ref:
#                 train_images_dir = '/'.join(v.ref[7:].split('/')[:-1])
#                 train_images_list.append(v.ref.split('/')[-1])
#     return train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list
