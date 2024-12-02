
## uploading the directory from a remote system to wandb####################3
import wandb
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="training_unet_allfeatures", name="upload_remote_test_data", id="saving_data", resume = "allow")
artifact = wandb.Artifact("test-dataset", type="dataset")
artifact.add_dir("/home/danfossautonomypc/segmentation/test_data/")
wandb.log_artifact(artifact)

wandb.finish()