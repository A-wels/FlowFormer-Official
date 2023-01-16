import sys

import imageio
from tqdm import tqdm
sys.path.append('core')

from PIL import Image
from glob import glob
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.submission import get_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils.frame_utils import read_gen
import cv2
import math
import os.path as osp

from core.FlowFormer import build_flowformer

from utils.utils import InputPadder, forward_interpolate
import itertools

IMAGE_SIZE = [344,127]
def get_image(path):
        flow = read_gen(path)
        flow_img = flow_viz.flow_to_image(flow)     
        flow = np.transpose(flow, (1,0,2))
        flow_img = np.transpose(flow_img, (1,0,2))
        return flow, flow_img

def generate_vector_visualization(flow, flow_img,title,output_path, step=10):
    copy_image = flow_img.copy()
    ims=dict(cmap='Greys', vmax=0.4*copy_image.max(),vmin=0)
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(20, 20))

    axs.imshow(copy_image, **ims)
    axs.set_title('MVF {} with vectors'.format(title))
    axs.set_xlabel("front-back")
    axs.set_ylabel("head-feet")

    step = 10
    quiver_opts=dict(color='red', angles='xy',scale_units='xy', scale=1)
    x,y =np.meshgrid(np.arange(0,flow_img.shape[1],step), np.arange(0,flow_img.shape[0],step))
    part1 = flow[::step,::step,0]
    part2 = flow[::step,::step,1]
    axs.quiver(x,y,part1,part2, **quiver_opts)
    plt.savefig(output_path)

def create_gif(list_of_png,viz_root_dir,title):
    # create animated gif out of pngs
    images = []
    
    for filename in tqdm(list_of_png, desc="Creating gif"):
        images.append(imageio.v2.imread(filename))
    imageio.mimsave(os.path.join(viz_root_dir,'{}.gif'.format(title)), images, duration=0.5)
    print("Saving: ")
    print("{}".format(os.path.join(viz_root_dir,'{}.gif'.format(title)), images, duration=0.1))
    print("Created gif")

def visualize_flow(viz_root_dir,gt_dir):
    if not os.path.exists(viz_root_dir):
        os.makedirs(viz_root_dir)

    list_of_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".mvf")])
    list_of_png = []

    for flowfile in tqdm(list_of_files, desc="Visualizing flow"):
        flow, flow_img = get_image(os.path.join(gt_dir,flowfile))
        output_path = os.path.join(viz_root_dir,flowfile.replace(".mvf", ".png"))


        generate_vector_visualization(flow, flow_img,flowfile, output_path)
        list_of_png.append(output_path)
    create_gif(list_of_png,viz_root_dir, gt_dir.split("/")[-1])

    
    


    
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/home/alex/Development/master/FlowFormer-Official/datasets/pet/validation/flow/000203_142')
    parser.add_argument('--viz_root_dir', default='viz_results')

    args = parser.parse_args()

    viz_root_dir = args.viz_root_dir
    gt_dir = args.gt_dir


   

    with torch.no_grad():
        visualize_flow(viz_root_dir,gt_dir)