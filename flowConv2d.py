import argparse
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
from torch.utils.tensorboard import SummaryWriter
from torch.nn import init

# https://github.com/NVIDIA/flownet2-pytorch/issues/60
# padding for input data
class padding(nn.Module):
    def __init__(self):
        super(padding,self).__init__()
        self.wpad = nn.ReplicationPad2d((0, -1, 0, 0))
        self.hpad = nn.ReplicationPad2d((0, 0, 0, -1))
    def forward(self, input, targetsize):
        if input.size()[2] != targetsize[2]:
            input = self.hpad(input)
        if input.size()[3] != targetsize[3]:
            input = self.wpad(input)
        return input
# Based on FlowNet2S: https://arxiv.org/abs/1612.01925
# https://github.com/NVIDIA/flownet2-pytorch/blob/master/networks/FlowNetS.py
'''
Portions of this code copyright 2017, Clement Pinard
'''
def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True)
    )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1,inplace=True)
    )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


class FlowNetS(nn.Module):
    def __init__(self):
        super(FlowNetS, self).__init__()
        print("Building model...")
        # padding for input data: divisible by 64
        self.pad = padding()
        self.conv1   = conv( 2,   64, kernel_size=7, stride=2)
        self.conv2   = conv( 64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(256,  256)
        self.conv4   = conv(256,  512, stride=2)
        self.conv4_1 = conv(512,  512)
        self.conv5   = conv(512,  512, stride=2)
        self.conv5_1 = conv(512,  512)
        self.conv6   = conv(512, 1024, stride=2)
        self.conv6_1 = conv(1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        # init_deconv_bilinear(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, x):

        out_conv1 = self.conv1(x)

        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        # pad
        flow6_up = self.pad(flow6_up, out_conv5.size())
        out_deconv5 = self.pad(out_deconv5, out_conv5.size())

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        # pad
        flow5_up = self.pad(flow5_up, out_conv4.size())
        out_deconv4 = self.pad(out_deconv4, out_conv4.size())

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        # pad
        flow4_up = self.pad(flow4_up, out_conv3.size())
        out_deconv3 = self.pad(out_deconv3, out_conv3.size())

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        # pad
        flow3_up = self.pad(flow3_up, out_conv2.size())
        out_deconv2 = self.pad(out_deconv2, out_conv2.size())

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            # upsample output with a factor of 4
            flow2 = self.upsample1(flow2)
            flow3 = self.upsample1(flow3)
            flow4 = self.upsample1(flow4)
            flow5 = self.upsample1(flow5)
            flow6 = self.upsample1(flow6)
            return flow2,flow3,flow4,flow5,flow6
        else:
            flow2 = self.upsample1(flow2)
            return flow2

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
            flow = torch.from_numpy(flow).float() 

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
            input_images = torch.stack([img1,img2], dim=0)
            flow = flow.permute(2, 0, 1)
            return input_images, flow, valid.float()

def validate(model):
    model.eval()
    dataset = OpticalFlowDataset(split='validation')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, )
    epe_list = []
    criterion = EPELoss()
    with torch.no_grad():
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]            
            output = model(inputs)
            # use the first output for loss
            output_cropped = outputs[0][:,:,0:344,0:127]
            epe = criterion(output_cropped, targets)
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
    return epe, px1, px3, px5

 # main method
if __name__ == '__main__':
    # Create arguments parser and parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--c', type=bool, default=False, help='load from checkpoint')
    args = parser.parse_args()

    # Create a summary writer
    writer = SummaryWriter(log_dir="logs")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define model
    model = FlowNetS().to(device)
    
    model = nn.DataParallel(model, [0,1])

    # load from checkpoint if specified
    if args.c:
        model.load_state_dict(torch.load("optical_flow_2d.pt"))

   # model = model.to(device)
    model = model.cuda()
    model.train()

    # Define a loss function
    #criterion = nn.MSELoss()
    criterion = EPELoss()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Load your data into a numpy array of shape (num_samples, 2, 344, 344, 172)


    # Create the dataset
    dataset = OpticalFlowDataset()

    # Create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, )

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # Train the model
    epoch_num = args.epochs 
    for epoch in range(epoch_num):
        print("Starting training epoch {} out of {}".format(epoch+1, epoch_num))
        for i, data_blob in tqdm(enumerate(dataloader), total=len(dataloader)):
            (inputs, targets,_ ) = [x.cuda() for x in data_blob]
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            output_cropped = outputs[0][:,:,0:344,0:127]
            loss = criterion(output_cropped, targets)
            # Write log
            writer.add_scalar("Loss/train", loss, epoch)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
        # Print the loss every X epochs
        if (epoch+1) % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch_num, loss.item()))

        # Validate every X epochs
        if (epoch+1)%100 == 0:
            epe, px1, px3, px5 = validate(model)
            model.train()
            writer.add_scalar("EPE", epe, epoch)
            writer.add_scalar("px1", px1, epoch)
            writer.add_scalar("px3", px3, epoch)
            writer.add_scalar("px5", px5, epoch)
            
        if (epoch+1) % 350 == 0:
            torch.save(model.state_dict(), 'optical_flow_2d_{}.pt'.format(epoch+1))

    # Save the model
    writer.flush()

