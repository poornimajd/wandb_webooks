import wandb
# import cv2
import requests
# import wandb
import numpy as np
from PIL import Image
import io
import cv2


####################################################### if u want to add any files
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="training_unet_allfeatures", name="uploading_reference_data", id="ref_upload", resume = "allow")

artifact = run.use_artifact('poornima-dharamdasani-danfoss/training_unet_allfeatures/segment_dataset_reference:v0', type='dataset')
# wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))

draft_artifact = artifact.new_draft()  # create a draft version

# modify a subset of files in the draft version
draft_artifact.add_file("/home/danfossautonomypc/segmentation/readme_upload.txt")
# draft_artifact.remove("dir_to_remove/")
run.log_artifact(
    draft_artifact
) 