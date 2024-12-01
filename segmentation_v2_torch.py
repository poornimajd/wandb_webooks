import torch
import os
from torch import cuda
# from srcutils import parse_code
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from src.model import *
from src.loss import FocalLoss
from src.utils import *
from src.Camvid import *
import pandas as pd
from src.eval import *
from src.config import config
import wandb
from wandb import AlertLevel
# from increment_refdata_use import get_data

wandb.login()

# run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="training_unet_version2", id="wandb_unet_v2", resume = "allow")
run = wandb.init(settings=wandb.Settings(start_method="fork"), project="segment", name="training_unet_version3", id="wandb_unet_v3")

wandb.run.log_code(".")  ########## wandb will capture all python source code files in the current and all 
# sub directories as an artifact. 
# wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
artifact = run.use_artifact('poornima-dharamdasani-danfoss/segment/segment_dataset_reference:v1', type='dataset') #this has already been uploaded so u can use it.
CONFIG = config()
# config.get_artifact(artifact)

# path = CONFIG.path
# train_images_dir =CONFIG.train_images_dir
# train_masks_dir = CONFIG.train_masks_dir
# val_images_dir = CONFIG.val_images_dir
# val_masks_dir = CONFIG.val_masks_dir
batch = CONFIG.batch
lr = CONFIG.lr
epochs = CONFIG.epochs
device = CONFIG.device
print(f"The device being used is: {device}\n")
# id2code = CONFIG.id2code
input_size = CONFIG.input_size
model_sv_pth = CONFIG.model_path
print("Changed the code!!!!!!!!!!")

##log all the parameters neeeded
wandb.config.update({"learning_rate": lr, "batch_size": batch, "epochs": epochs, "input_size":input_size},allow_val_change=True) #there are different ways in configure experiments.


def get_data():
    train_images_list = []
    val_images_list=[]
    train_masks_list=[]
    val_masks_list=[]
    val_masks_dir = None
    train_masks_dir = None
    val_images_dir = None
    train_images_dir =None
    for k,v in artifact.manifest.entries.items():
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

train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list = get_data()

source_folder = os.path.dirname(os.path.dirname(train_images_dir))
label_codes, label_names = zip(*[parse_code(l) for l in open(source_folder+"/label_colors.txt")])
label_codes, label_names = list(label_codes), list(label_names)

code2id = {v:k for k,v in enumerate(label_codes)}
id2code = {k:v for k,v in enumerate(label_codes)}
# print("ddddddddd",id2code)

name2id = {v:k for k,v in enumerate(label_names)}
id2name = {k:v for k,v in enumerate(label_names)}


def train(model, trainloader, validloader, criterion, optimizer, epochs, device, model_sv_pth, plot=True, visualize=True, load_model=False):
  model.train()
  stats = []
  valid_loss_min = np.Inf
  print('Training Started.....')


  for epoch in range(epochs):
    
    train_loss=0
    train_iou = []
    for i, data in enumerate(trainloader):
      inputs, mask, rgb = data
      inputs, mask = inputs.to(device), mask.to(device)

      optimizer.zero_grad()
      outputs=model(inputs.float())
      
      labels = mask.argmax(1)
      loss = criterion(outputs, labels.long())
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * inputs.size(0) 
      iou = iou_pytorch(outputs.argmax(1), labels)


      
      train_iou.extend(iou)     
      ##########################instead of below, use wandb to visualize
      # if visualize and epoch%10==0 and i == 0:
      #   print('The training images')
      #   show_databatch(inputs.detach().cpu(), size=(8,8))
      #   print('The original masks')
      #   show_databatch(rgb.detach().cpu(), size=(8,8))
      #   RGB_mask =  onehot_to_rgb(outputs.detach().cpu(), id2code)
      #   print('Predicted masks')
      #   show_databatch(torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
      if visualize and epoch%5==0 and i == 0:
        print('The training images')
        inputs_out = show_databatch(inputs.detach().cpu(), size=(8,8))
        inputs_out = inputs_out.numpy().transpose((1, 2, 0))
        images = wandb.Image(inputs_out, caption="Original image")

        wandb.log({"examples": images})

        print('The original masks')
        rgb_out = show_databatch(rgb.detach().cpu(), size=(8,8))
        rgb_out = rgb_out.numpy().transpose((1, 2, 0))
        rgb_out = wandb.Image(rgb_out, caption="original masks")

        wandb.log({"gtmasks": rgb_out})

        
        print('Predicted masks')
        RGB_mask =  onehot_to_rgb(outputs.detach().cpu(), id2code)
        RGB_mask= show_databatch(torch.tensor(RGB_mask).permute(0,3,1,2), size=(8,8))
        RGB_mask = RGB_mask.numpy().transpose((1, 2, 0))
        RGB_mask = wandb.Image(RGB_mask, caption="predicted masks")

        wandb.log({"predmasks":RGB_mask})


      miou = torch.FloatTensor(train_iou).mean()
      # sending alerts usinf wandb
      if miou<0.1:
          run.alert(
              title="Low miou",
              text=f"miou {miou} is below the acceptable threshold {0.1}",
              level=AlertLevel.WARN,
              wait_duration=100,
          )
      train_loss = train_loss / len(trainloader.dataset)
      print('Epoch',epoch,':',f'Lr ({optimizer.param_groups[0]["lr"]})',f'\n\t\t Training Loss: {train_loss:.4f},',f' Training IoU: {miou:.3f},')
      wandb.log({"loss": train_loss, "miou": miou})

    with torch.no_grad():
      valid_loss, valid_loss_min = Validate(model, validloader, criterion, valid_loss_min, device, model_sv_pth,epoch)
      wandb.log({"valid_loss": valid_loss})
    # Log the model
    file_path = model_sv_pth+'/state_dict'+f'{str(epoch)}'+'s.pt'
    if os.path.exists(file_path):
      # run.log_model(path=file_path, name=f'{str(epoch)}'+'s.pt')
      logged_artifact = run.log_artifact(artifact_or_path=file_path, name=f'{str(epoch)}'+'s.pt', type="model") #logging as artifacts, you can link it thru the ui to the registry
      # run.link_artifact(artifact=logged_artifact, target_path=f"poornima-dharamdasani-danfoss-org/wandb-registry-model/registry-quickstart-collection"),
      # run.link_model(path=file_path, registered_model_name=registered_model_name)

  stats.append([train_loss, valid_loss])
  stat = pd.DataFrame(stats, columns=['train_loss', 'valid_loss'])

  print('Finished Training')
  if plot: plotCurves(stat) #it is plotted on wandb



if __name__ == "__main__":
    

    #pass transform here-in
    ##################################################################upload the data from the remote mount to registry as a reference artifact, use the data from wandb directly:
    # train_images_dir, val_images_dir, train_masks_dir, val_masks_dir, train_images_list, val_images_list, train_masks_list, val_masks_list = get_data(artifact)
    # train_images = 
    train_data = CamSeqDataset(image_dir = train_images_dir, mask_dir = train_masks_dir,images= train_images_list, masks = train_masks_list, image_size=input_size,id2code = id2code)
    valid_data = CamSeqDataset(image_dir = val_images_dir, mask_dir = val_masks_dir,images= val_images_list, masks = val_masks_list,image_size=input_size,id2code = id2code)


    #data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch, shuffle=True)

    # model = UNet(3, 32, True).to(device)
    model =UNet(n_filters=32, bn=True, dilation_rate=1)
    model = model.to(device)
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.99)

    train(model, trainloader, validloader, criterion, optimizer, epochs, device, model_sv_pth, plot=False, visualize=True, load_model=False)