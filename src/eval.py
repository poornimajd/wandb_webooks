import torch
import torch.nn as nn
import numpy as np
from src.utils import *
from src.config import config
# import wandb 

# CONFIG = config()
# id2code = CONFIG.id2code
def Validate(model, validloader, criterion, valid_loss_min, device, model_path,epoch):
	#validating
	model.eval()
	val_loss=0.0
	val_iou =[]
	val_losses=[]
	for i, data in enumerate(validloader):
		inputs, mask, _ = data
		inputs, mask = inputs.to(device), mask.to(device)

		outputs = model(inputs)

		labels = mask.argmax(1)
		loss=criterion(outputs, labels.long())
		val_loss+=loss.item()* inputs.size(0)
		preds=outputs.argmax(dim=1)
		iou = iou_pytorch(outputs.argmax(1), labels)
		val_iou.extend(iou)    
	miou = torch.FloatTensor(val_iou).mean()
	valid_loss = val_loss / len(validloader.dataset)

	val_losses.append(valid_loss)
	print(f'\t\t Validation Loss: {valid_loss:.4f},',f' Validation IoU: {miou:.3f}')

	if np.mean(val_losses) <= valid_loss_min:
	    torch.save(model.state_dict(), model_path+'/state_dict'+f'{str(epoch)}'+'s.pt')

		
	    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(valid_loss_min,np.mean(val_losses))+'\n')
	    valid_loss_min = np.mean(val_losses)

	return valid_loss, valid_loss_min


def Test_eval(model, testloader, criterion,device,id2code,model_dir):
	model.load_state_dict(torch.load(model_dir))
	model.eval()
	test_loss=0.0
	imgs, masks, preds=[],[],[]
	for i, data in enumerate(testloader):
		inputs, mask= data
		inputs, mask = inputs.to(device), mask.to(device)
		imgs.extend(inputs.cpu().numpy())
		masks.extend(mask.cpu().numpy())

		outputs = model(inputs.float())
		preds.extend(outputs.detach().cpu().numpy())
		labels = mask.argmax(1)
		loss=criterion(outputs, labels.long())
		test_loss+=loss.item()* inputs.size(0)
		
	test_loss = loss / len(testloader.dataset)
	pred = onehot_to_rgb(np.array(preds), id2code)
	print(f"Test loss is: {test_loss:.4f}")
	return np.array(imgs), np.array(masks), np.array(pred)
