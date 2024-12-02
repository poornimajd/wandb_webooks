import wandb
import os
# from wandb import AlertLevel
# from increment_refdata_use import get_data
##################################################################################WANDB###################################################
class WandbConfig:
    def __init__(self):

        wandb.login()

        self.run = wandb.init(settings=wandb.Settings(start_method="fork"), project="training_unet_allfeatures", name="training_unet", id="wandb_unet")

        # wandb.run.log_code(".")  ########## wandb will capture all python source code files in the current and all  sub directories as an artifact. 
        self.run.log_code(
            ".",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
            exclude_fn=lambda path, root: (os.path.relpath(path, root).startswith("data/") or os.path.relpath(path, root).startswith("new/"))
        )

        self.artifact = self.run.use_artifact('poornima-dharamdasani-danfoss/training_unet_allfeatures/segment_dataset_reference:v1', type='dataset') #this has already been uploaded so u can use it.

    def configure_experiment(self, config):
        wandb.config.update({"learning_rate": config.lr, "batch_size": config.batch, "epochs": config.epochs, "input_size":config.input_size},allow_val_change=True) #there are different ways in configure experiments.

    def get_data(self): # get from wandb directly, without downloading the data to local system
        train_images_list = []
        val_images_list=[]
        train_masks_list=[]
        val_masks_list=[]
        val_masks_dir = None
        train_masks_dir = None
        val_images_dir = None
        train_images_dir =None
        for k,v in self.artifact.manifest.entries.items():
        #   print(v.ref
            if v.ref is not None:
                if 'val_masks' in v.ref:
                    val_masks_dir = '/'.join(v.ref[7:].split('/')[:-1])
                    val_masks_list.append(v.ref.split('/')[-1])
                if 'train_masks' in v.ref:
                    train_masks_dir = '/'.join(v.ref[7:].split('/')[:-1])
                    train_masks_list.append(v.ref.split('/')[-1])
                if 'val_frames' in v.ref:
                    val_images_dir = '/'.join(v.ref[7:].split('/')[:-1])
                    val_images_list.append(v.ref.split('/')[-1])
                if 'train_frames' in v.ref:
                    train_images_dir = '/'.join(v.ref[7:].split('/')[:-1])
                    train_images_list.append(v.ref.split('/')[-1])
        return train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list

