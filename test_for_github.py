import torch
from torchvision import datasets, transforms
from src.model import UNet
from src.loss import FocalLoss
from src.config import config
from src.utils import *
from src.Camvid import *
from src.eval import *
import argparse
import os
import numpy as np

CONFIG = config()
# path = CONFIG.path
batch = CONFIG.batch
input_size = CONFIG.input_size
# load_model_pth = CONFIG.load_model
# device = CONFIG.device

import wandb
# from segmentation_v2_torch import get_data

# wandb.login()
wandb.login(key=os.getenv("WANDB_API_KEY"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run W&B testing workflow.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--model_artifact", type=str, required=True, help="Model artifact reference")
    parser.add_argument("--label_colors_path", type=str, required=True, help="Path to label colors file")
    args = parser.parse_args()
    def sanitize_name(name):
        """
        Remove all '-' characters from the given name.
        """
        return name.replace('-', '')
    project_name = sanitize_name(args.project)
    run = wandb.init(settings=wandb.Settings(start_method="fork"), project=project_name) #give these aas argumentys in workflow.
    run.log_code(
    ".",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: (os.path.relpath(path, root).startswith("wandb/") or os.path.relpath(path, root).startswith("data/"))
        )

    columns = ["id", "image", "gt masks", "pred masks"]
    test_table = wandb.Table(columns=columns)

    # 
    artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/test-dataset:v2', type='dataset')
    valid_dir = artifact.download()
    # print(valid_dir)
    val_images_dir = os.path.join(valid_dir,'frames')
    val_masks_dir =os.path.join(valid_dir,'masks')
    # print(val_masks_dir)

    artifact = run.use_artifact(args.model_artifact, type='model') #using registry for best model, after linking thru ui #give as argument
    model_dir = artifact.download()

    model_dir=os.path.join(model_dir,os.listdir(model_dir)[0]) #as argument
    
    valid_data = Test(image_dir = val_images_dir, mask_dir = val_masks_dir,image_size=input_size)

    testloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    model =UNet(n_filters=32, bn=True, dilation_rate=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model = model.to(device)
    criterion = FocalLoss()


    label_codes, label_names = zip(*[parse_code(l) for l in open(args.label_colors_path)]) #give path relative to github
    label_codes, label_names = list(label_codes), list(label_names)

    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}

    imgs, masks, pred = Test_eval(model, testloader, criterion, device,id2code,model_dir)
    num_images = imgs.shape[0]

    for img_id in range(num_images):
        img = np.transpose(imgs[img_id], (1, 2, 0))
        gt_mask = np.transpose(masks[img_id],(1,2,0))
        pred_mask = pred[img_id]
        # img_channels_last = 
        test_table.add_data(img_id, wandb.Image(img), wandb.Image(gt_mask), wandb.Image(pred_mask))
    artifact = wandb.Artifact("test_table_results", type="table")
    # artifact.add(test_table)
    artifact.add(test_table, "test_results_table")
    run.log_artifact(artifact)
