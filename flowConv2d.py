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
     
batch_size = 5
import torch.nn as nn

class OpticalFlow2D(nn.Module):
    def __init__(self):
        super(OpticalFlow2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 43 * 43, 128 * 15 * 43 * 2)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(128 * 15 * 43, 344 * 127 * 2)
        #self.fc = nn.Linear(128, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
     #   print("after conv: {}".format(x.shape))
        # flatten input from convolutional layers and pass throug fc layers
        x = x.view(-1, 412800)        
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        # reshape for output
        x = x.view(-1, 2, 127,344)
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
            flow = torch.from_numpy(flow).permute(2, 0, 1).float() 

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            input_images = torch.stack([img1,img2], dim=0)
            return input_images, flow, valid.float()

 # main method
if __name__ == '__main__':
        
    # Define your model
    model = OpticalFlow2D()
    model = model.cuda()
    model = nn.DataParallel(model, [0])
    model.train()

    # Define a loss function
    criterion = nn.MSELoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Load your data into a numpy array of shape (num_samples, 2, 344, 344, 172)


    # Create the dataset
    dataset = OpticalFlowDataset()

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )

    # Define a loss function
    criterion = nn.MSELoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epoch_num = 1000
    for epoch in range(epoch_num):
        for i, data_blob in enumerate(dataloader):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)


            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Print the loss every 20 epochs
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))
            print(outputs[0].min())
            print(outputs[0].max())
            print("----------------------")
            print(targets[0].min())
            print(targets[0].max())
            print("\n")
    # Save the model
    torch.save(model.state_dict(), 'optical_flow_2d.pt')
