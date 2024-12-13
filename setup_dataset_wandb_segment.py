# adding only a reference###################
import wandb
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="uploading_mount_data", id="mount_upload", resume = "allow")
# wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
artifact = wandb.Artifact("segment_dataset_reference", type="dataset")
artifact.add_reference("file:///home/danfossautonomypc/segmentation/mount/system_b/segment_data/CamSeq01/")
# Reference files in the mounted directory

run.log_artifact(artifact)

########################### you can link it to collection or registry via GUI