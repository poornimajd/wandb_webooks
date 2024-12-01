
## uploading the directory from a remote system to wandb####################3
import wandb
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="upload_remote_data", id="saving_data", resume = "allow")
# wandb.init(project="your_project_name")
# sftp://danfossautonomypc@10.24.2.200/home/danfossautonomypc/segmentation/mount/system_b
# Reference files in the mounted directory
artifact = wandb.Artifact("test-dataset", type="dataset")
artifact.add_dir("../test_data/")
wandb.log_artifact(artifact)

wandb.finish()