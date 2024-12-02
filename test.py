import torch
from torchvision import datasets, transforms
from src.model import UNet
from src.loss import FocalLoss
from src.config import config
from src.utils import *
from src.Camvid import *
from src.eval import *
import argparse
import wandb



if __name__ == "__main__":


    CONFIG = config()
    # path = CONFIG.path
    batch = CONFIG.batch
    input_size = CONFIG.input_size

    parser = argparse.ArgumentParser(description="Run W&B testing workflow.")
    parser.add_argument("--project", type=str, required=True, help="W&B project name")
    parser.add_argument("--name", type=str, required=True, help="Run name for W&B")
    parser.add_argument("--id", type=str, required=True, help="Run ID for W&B")
    parser.add_argument("--resume", type=str, required=True, help="Resume mode for W&B")
    parser.add_argument("--model_artifact", type=str, required=True, help="Model artifact reference")
    parser.add_argument("--label_colors_path", type=str, required=True, help="Path to label colors file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The device being used is: {device}\n")
    ##################################################################################WANDB###################################################
    wandb.login()

    run = wandb.init(settings=wandb.Settings(start_method="fork"), project=args.project, name=args.name, id=args.id, resume=args.resume)


    run.log_code(
    ".",
    include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"),
    exclude_fn=lambda path, root: (os.path.relpath(path, root).startswith("wandb/") or os.path.relpath(path, root).startswith("data/"))
        )

    
    columns = ["id", "image", "gt masks", "pred masks"]
    test_table = wandb.Table(columns=columns)
    
    artifact = run.use_artifact('poornima-dharamdasani-danfoss/training_unet_allfeatures/test-dataset:v0', type='dataset')
    valid_dir = artifact.download()
    
    val_images_dir = valid_dir+'/frames/'
    val_masks_dir =valid_dir+'/masks/'

    artifact = run.use_artifact(args.model_artifact, type='model') #using registry for best model, after linking thru ui #give as argument
    model_dir = artifact.download()
    ##################################################################################WANDB###################################################
    model_dir=os.path.join(model_dir,os.listdir(model_dir)[0]) 

   
    valid_data = Test(image_dir = val_images_dir, mask_dir = val_masks_dir,image_size=input_size)

    testloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    # model = UNet(3, 32, True).to(device)
    model =UNet(n_filters=32, bn=True, dilation_rate=1)
    model = model.to(device)
    criterion = FocalLoss()

    
    label_codes, label_names = zip(*[parse_code(l) for l in open(args.label_colors_path)]) #give path relative to github
    label_codes, label_names = list(label_codes), list(label_names)

    code2id = {v:k for k,v in enumerate(label_codes)}
    id2code = {k:v for k,v in enumerate(label_codes)}

    imgs, masks, pred = Test_eval(model, testloader, criterion, device,id2code,model_dir)
    num_images = imgs.shape[0]
    # print(num_images)
    print(imgs.shape, masks.shape, pred.shape)
    for img_id in range(num_images):
        img = np.transpose(imgs[img_id], (1, 2, 0))
        gt_mask = np.transpose(masks[img_id],(1,2,0))
        pred_mask = pred[img_id]

        ##################################################################################WANDB###################################################
        test_table.add_data(img_id, wandb.Image(img), wandb.Image(gt_mask), wandb.Image(pred_mask))
    run.log({"table_results": test_table})

