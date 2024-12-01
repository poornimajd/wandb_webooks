import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torchvision



def parse_code(l):
    '''Function to parse lines in a text file, returns separated elements (label codes and names in this case)
    '''
    if len(l.strip().split("\t")) == 2:
        a, b = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), b
    else:
        a, b, c = l.strip().split("\t")
        return tuple(int(i) for i in a.split(' ')), c

def rgb_to_onehot(rgb_image, colormap):
    '''Function to one hot encode RGB mask labels
        Inputs: 
            rgb_image - image matrix (eg. 256 x 256 x 3 dimension numpy ndarray)
            colormap - dictionary of color to label id
        Output: One hot encoded image of dimensions (height x width x num_classes) where num_classes = len(colormap)
    '''
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.numpy()
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.float64 )
    for cls_id, class_color in colormap.items():
        color_array = np.array(class_color).reshape(1, 1, 3)
        # print("dsssssssssssss",color_array,rgb_image.shape)
        mask = np.all(rgb_image == color_array, axis=-1)
        # print(mask.shape)
        encoded_image[:, :, cls_id] = mask

    # for i, cls in enumerate(colormap):
        # encoded_image[:,:,cls_id] = np.all(rgb_image == np.array(class_color).reshape(1, 1, 3), axis=-1)
    return encoded_image.transpose(2,0,1)

def onehot_to_rgb(onehot, colormap):
    '''Function to decode encoded mask labels
        Inputs: 
            onehot - one hot encoded image matrix (height x width x num_classes)
            colormap - dictionary of color to label id
        Output: Decoded RGB image (height x width x 3) 
    '''
    onehot = onehot.numpy() if isinstance(onehot, torch.Tensor) else onehot
    batch_size, num_classes, height, width = onehot.shape

    # Initialize output array with shape (batch_size, height, width, 3)
    output = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    
    # Get the single-layer mask for each pixel (most probable class)
    single_layer = np.argmax(onehot, axis=1)  # Shape: (batch_size, height, width)

    for k, color in colormap.items():
        output[single_layer == k] = color
    
    return np.uint8(output)

SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded

def plotCurves(stats):
    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(stats[c], label=c)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Training and Validation Loss Curve')
    plt.show()

def Visualize(imgs, title='Original', cols=6, rows=1, plot_size=(16, 16), change_dim=False):
    fig=plt.figure(figsize=plot_size)
    columns = cols
    rows = rows
    print(rows, columns)
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.title(title+str(i))
        if change_dim: plt.imshow(imgs.transpose(0,2,3,1)[i])
        else: plt.imshow(imgs[i])
    plt.show()

def imshow(inp, size, title=None):
    '''
        Shows images

        Parameters:
            inp: images
            title: A title for image
    '''
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=size)
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, size=(8,8), batch=16):
    out = torchvision.utils.make_grid(inputs[:batch])
    return out
    # imshow(out, size)