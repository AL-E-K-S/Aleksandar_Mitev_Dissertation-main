from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os

# Code copied from Koot., R., E.(2021.) Efficient Video Fataset Loading and Augmentation, GitHub.   https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch  

 
from torch.utils.data import DataLoader, Dataset

# code from: reference 
# 
videos_root = os.path.join(os.getcwd(), 'Real Life Violence Dataset')
annotation_file = os.path.join(videos_root, 'annotations.txt')

preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Grayscale(),
        transforms.Normalize((0.5, ), (0.5,)), # normalizes the images 
        transforms.Resize((64,64))  # image batch, resize smaller edge to 299  
        
    ])

dataset = VideoFrameDataset(
    root_path=videos_root, # location of the folder where the folders with frames are
    annotationfile_path=annotation_file, # location of the file where the lebels of teh videos are
    num_segments=1, # number of segments into which the video will be divided into 
    frames_per_segment=30, # number of frames each segment of the video will have 
    imagefile_template="img_{:05d}.jpg", # format in which the images are saved 
    transform=preprocess, # sets an option to do the preprocessing > transforms done on the frames of the videos
    test_mode=False 
)
