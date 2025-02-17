import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from flowConv2d import FlowNetS
from flowConv3d import OpticalFlow3D
from utils import flow_viz
import cv2
from utils import frame_utils
from visualize_gt_flow import create_gif, generate_vector_visualization
import os

from utils.frame_utils import read_gen

def load_model(model_path):
    model = FlowNetS()
    model = nn.DataParallel(model, [0,1])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# load the demo images from the given path
def load_demo_data(input_path):
    list_of_files = os.listdir(input_path)
    list_of_files.sort()
    file_names = []
    for file in list_of_files:
        p = os.path.join(input_path, file)
        if p.endswith(".v"):
            file_names.append(p)
    
    vector_fields = []
    for i in range(len(file_names)-1):
        img1 = read_gen(file_names[i])
        img2 = read_gen(file_names[i+1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()

        vector_fields.append([img1, img2])
    return vector_fields



if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='optical_flow_2d.pt', help='path to the model')
    parser.add_argument('--input', type=str, default='demo_data', help='path to the input')
    parser.add_argument('--output', type=str, default='viz_results', help='path to the output')
    args = parser.parse_args()
   # load model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    # load model
    model = load_model(args.model_path).to(device)
    # set model for inference
    model.eval()

    # check if model is running on GPU
    if next(model.parameters()).is_cuda:
        print("Model is running on GPU")
    else:
        print("Model is running on CPU")

    print("Loaded model from {}".format(args.model_path))
    
    # load demo data
    vector_fields = load_demo_data(args.input)

    # move demo data to GPU if available
    for i in range(len(vector_fields)):
        vector_fields[i][0] = vector_fields[i][0].to(device)
        vector_fields[i][1] = vector_fields[i][1].to(device)

    list_of_images = []
    # predict
    for i in tqdm(range(len(vector_fields))):
        img1 = vector_fields[i][0]
        img2 = vector_fields[i][1]
        input_images = torch.stack([img1,img2], dim=0)
        # add dimension for batch size
        input_images = input_images.unsqueeze(0)
        prediction = model(input_images)
        # remove added dimension for batch size
        prediction = prediction.squeeze(0).cpu().detach().numpy()

        prediction = np.transpose(prediction, (2,1,0))

        frame_utils.writeFlow(os.path.join(args.output, 'flow_{}.flo'.format(i+2)),prediction)
        flow_img = flow_viz.flow_to_image(prediction)
        output_path = os.path.join(args.output, 'flow_{}.png'.format(i))
        cv2.imwrite(output_path, flow_img[:, :, :])
        generate_vector_visualization(prediction, flow_img, "flow{}".format(i+2), output_path)
        list_of_images.append(output_path)

    create_gif(list_of_images,args.output, list_of_images[0].split("/")[-1].replace('.png', ''))

    

