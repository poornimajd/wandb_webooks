import torch
from torchvision import datasets, transforms
from src.model import UNet
from src.loss import FocalLoss
from src.config import config
from src.utils import *
from src.Camvid import *
from src.eval import *

CONFIG = config()
# path = CONFIG.path
batch = CONFIG.batch
input_size = CONFIG.input_size
# load_model_pth = CONFIG.load_model
device = CONFIG.device
# train_images_dir =CONFIG.train_images_dir
# train_masks_dir = CONFIG.train_masks_dir
# val_images_dir = CONFIG.val_images_dir
# val_masks_dir = CONFIG.val_masks_dir
import wandb
# from segmentation_v2_torch import get_data

wandb.login()

if __name__ == "__main__":
    run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="test model", id="testing", resume="allow")

    ref_artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/segment_dataset_reference:v1', type='dataset')

    def get_data():
        train_images_list = []
        val_images_list=[]
        train_masks_list=[]
        val_masks_list=[]
        val_masks_dir = None
        train_masks_dir = None
        val_images_dir = None
        train_images_dir =None
        for k,v in ref_artifact.manifest.entries.items():
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

    #pass transform here-in
    # train_data = Test(image_dir = train_images_dir, mask_dir = train_masks_dir,image_size=input_size)
    columns = ["id", "image", "gt masks", "pred masks"]
    test_table = wandb.Table(columns=columns)

    # 
    artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/test-dataset:v2', type='dataset')
    valid_dir = artifact.download()
    # print(valid_dir)
    val_images_dir = valid_dir+'/frames/'
    val_masks_dir =valid_dir+'/masks/'
    # print(val_masks_dir)

    artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/8s.pt:v1', type='model') #using registry for best model, after linking thru ui
    model_dir = artifact.download()
    model_dir+='/state_dict8s.pt'
    #data loaders
    # trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    valid_data = Test(image_dir = val_images_dir, mask_dir = val_masks_dir,image_size=input_size)

    testloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    # model = UNet(3, 32, True).to(device)
    model =UNet(n_filters=32, bn=True, dilation_rate=1)
    model = model.to(device)
    criterion = FocalLoss()

    train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list = get_data()
    
    source_folder = os.path.dirname(os.path.dirname(train_images_dir))
    label_codes, label_names = zip(*[parse_code(l) for l in open(source_folder+"/label_colors.txt")])
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
        # img_channels_last = 
        test_table.add_data(img_id, wandb.Image(img), wandb.Image(gt_mask), wandb.Image(pred_mask))
    run.log({"table_results": test_table})












    #     print(img_id)
    # 

    # Visualize(imgs, 'Original Image', 6, 1, change_dim=True)
    # Visualize(masks, 'Original Mask', 6, 1, change_dim=True)
    # Visualize(pred, 'Predicted mask', 6, 1, change_dim=False)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    # train(model, trainloader, validloader, criterion, optimizer, epochs, device, model_sv_pth, plot=True, visualize=False, load_model=False)



##to try : add the reference on wandb, and then use the data from that referene for downstream tasks, can be model or datasetr.

### uploading the directory from a remote system to wandb####################3
# import wandb
# run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="experiment_2", id="saving external artifacts", resume = "allow")
# # wandb.init(project="your_project_name")
# # sftp://danfossautonomypc@10.24.2.200/home/danfossautonomypc/segmentation/mount/system_b
# # Reference files in the mounted directory
# artifact = wandb.Artifact("my-remote-dataset", type="dataset")
# artifact.add_dir("./mount/system_b/CamSeq01/")
# wandb.log_artifact(artifact)

# wandb.finish()

#anoth er script to add files and increment ,
# in this file download it and use
