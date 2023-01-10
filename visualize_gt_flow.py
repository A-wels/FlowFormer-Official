import sys
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

def visualize_flow(viz_root_dir,gt_dir):
    if not os.path.exists(viz_root_dir):
        os.makedirs(viz_root_dir)

    for flowfile in [f for f in os.listdir(gt_dir) if f.endswith(".mvf")]:
        flow = read_gen(os.path.join(gt_dir,flowfile))

        flow_img = flow_viz.flow_to_image(flow)
        output_path = os.path.join(viz_root_dir,flowfile.replace(".mvf", ".png"))
        print(output_path)
        cv2.imwrite(output_path, np.swapaxes(flow_img, 0, 1))

   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='/home/alex/Development/master/FlowFormer-Official/datasets/pet/validation/flow/000001_142')
    parser.add_argument('--viz_root_dir', default='viz_results')

    args = parser.parse_args()

    viz_root_dir = args.viz_root_dir
    gt_dir = args.gt_dir


   

    with torch.no_grad():
        visualize_flow(viz_root_dir,gt_dir)