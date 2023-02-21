import random
import sys

from tqdm import tqdm
sys.path.append('core')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from configs.pet import get_cfg
from datasets import PET
from glob import glob
import os.path as osp
from utils.frame_utils import read_gen
from myutils.EPELoss import EPELoss
batch_size = 128
import torch.nn as nn
import torch.nn.functional as F

# Based on FlowNet2: https://arxiv.org/abs/1612.01925
def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
        nn.LeakyReLU(0.1, inplace=True)
    )
class OpticalFlow2D(nn.Module):
    def __init__(self):
        super(OpticalFlow2D, self).__init__()
        print("Building model...")
        self.conv1 = conv(2, 64, 3, 1)
        self.conv2 = conv(64, 128, 3, 2)
        self.conv3 = conv(128, 256, 3, 2)
        self.conv4 = conv(256, 512, 3, 2)
        self.conv5 = conv(512, 512, 3, 2)

        self.fc1 = nn.Linear(512 * 22 * 8, 512)

        self.drop = nn.Dropout2d(0.2)

        self.fc2 = nn.Linear(512, 2 * 127 * 344)
        # define attention layer
      #  self.attention = nn.Conv2d(512, 1, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #print(x.shape)
        # compute attention weights for each spatial location
      #  attention_weights = self.attention(x)
        # reshape attention weights to match feature map dimensions
       # x = x * attention_weights

       # attention_weights = attention_weights.view(-1, 1, x.shape[2], x.shape[3])
        # apply attention weights to feature maps
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        # flatten input from convolutional layers and pass through fc layers
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)

        # reshape for output
        x = x.view(-1, 2, 127, 344)
        return x

# Define a custom dataset class
class OpticalFlowDataset(torch.utils.data.Dataset):
    def __init__(self, split='training', root='datasets/pet'):
        self.image_list = []
        self.flow_list = []
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, 'clean')

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.v')))
            for i in range(len(image_list)-1):
                self.image_list += [[image_list[i], image_list[i+1]]]

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root,
                                            scene, '*.mvf')))     
        print("Traning data size: ", len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
            index = index % len(self.image_list)
            valid = None
            flow = read_gen(self.flow_list[index])

            img1 = read_gen(self.image_list[index][0])
            img2 = read_gen(self.image_list[index][1])

            flow = np.array(flow).astype(np.float32)
            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            # grayscale images
          #  if len(img1.shape) == 2:
          #      print("grayscale images ")
          #      img1 = np.tile(img1[..., None], (1, 1, 3))
          #      img2 = np.tile(img2[..., None], (1, 1, 3))
            #img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            #img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()
            flow = torch.from_numpy(flow).permute(2, 1, 0).float() 

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            input_images = torch.stack([img1,img2], dim=0)
            return input_images, flow, valid.float()


def validate():
    model.eval()
    dataset = OpticalFlowDataset(split='validation')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, )
    epe_list = []
    criterion = EPELoss()
    with torch.no_grad():
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]            
            output = model(inputs)
            epe = criterion(output, targets)
            epe_list.append(epe.detach().cpu())
    print(len(epe_list))
    epe = np.mean(epe_list)
    # px1: percentage of pixels with EPE < 1
    px1 = np.mean(np.array(epe_list) < 1)
    # px3: percentage of pixels with EPE < 3
    px3 = np.mean(np.array(epe_list) < 3)
    # px5: percentage of pixels with EPE < 5
    px5 = np.mean(np.array(epe_list) < 5)

    print("EPE: {:.3f}, px1: {:.3f}, px3: {:.3f}, px5: {:.3f}".format(epe, px1, px3, px5))
 # main method
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define your model
    model = OpticalFlow2D().to(device)
    #model = OpticalFlow2D()
    #model.load_state_dict(torch.load("optical_flow_2d.pt"))
    model = model.to(device)
    model = model.cuda()
    model = nn.DataParallel(model, [0,1])
    model.train()

    # Define a loss function
    #criterion = nn.MSELoss()
    criterion = EPELoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Load your data into a numpy array of shape (num_samples, 2, 344, 344, 172)


    # Create the dataset
    dataset = OpticalFlowDataset()

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Train the model
    epoch_num = 15
    for epoch in range(epoch_num):
        print("Starting training epoch {} out of {}".format(epoch+1, epoch_num))
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print the loss every X epochs
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))
            print(outputs[0].min())
            print(outputs[0].max())
            print("----------------------")
            print(targets[0].min())
            print(targets[0].max())
            print("\n")
    # Save the model
    validate()
    torch.save(model.state_dict(), 'optical_flow_2d.pt')
