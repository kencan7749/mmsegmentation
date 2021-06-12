import mmcv
import matplotlib.pyplot as plt

import numpy as np 
import os
import sys
from PIL import Image
from tqdm import tqdm


def create_mmseg_labels(label_root_dir, label_list,image_list,save_root_dir):
    """
    Created and save labels for mmsegmentation.
    Loaded label images (ranged 0-255) then regenerate labels accorging to their values (ranged 0 -num_classes -1).
    For rain_filtering dataset,
    0... nothing
    1... particle (based on first images)
    2... object (based on last images)
    3... particle and object (intersect between first and last)
    The label images are saved as png file as 'P' mode.
    
    Arguments:
        label_root_dir: path to directory that contains the label image direcotry
        label_list: list that contatins which label directory to use for labels
        image_list; list that contains which image file to create labels
        save_root_dir: path to directory that save directory
    """
    for img_file in tqdm(image_list):
        #load each images 
        label_image = create_label(label_root_dir, label_list, img_file)
        #convert to PIL.Image
        label_pil = Image.fromarray(label_image.astype(np.uint8))
        #save label image
        label_pil.save(os.path.join(save_root_dir, img_file), mode='P')

    print('done')
    

def create_label(label_root_dir, label_list, img_file):
    """
    Created and  labels for mmsegmentation.
    Loaded label images (ranged 0-255) then regenerate labels accorging to their values (ranged 0 -num_classes -1).
    Arguments:
        label_root_dir: path to directory that contains the label image direcotry
        label_list: list that contatins which label directory to use for labels
        img_file: str for load image file name
    Return: 
        img_array: np.array: image shape, whose values are ranged  (ranged 0 -num_classes -1).
        For rain_filtering datasets,
        0... nothing
        1... particle (based on first images)
        2... object (based on last images)
        3... particle and object (intersect between first and last)
    """
    lbl_img_list = []
    for label in label_list:
        #load image as float32
        lbl_img = mmcv.imread(os.path.join(label_root_dir, label, img_file)).astype(np.float32)
        #convert 255-> 1 (Note that pixel values are eigher 0 or 255)
        lbl_img /= 255
        # sum up the channel e.g. (height, width, 3) -> (height, width)
        lbl_img = lbl_img.sum(2)
        #convert3 -> 1 
        lbl_img /=3
        #append to list 
        lbl_img_list.append(lbl_img)
    #regenerate lbl_img
    img_array = np.zeros_like(lbl_img)
    for i, img in enumerate(lbl_img_list):
        img_array += img * (i+1)
    return img_array


if __name__ == '__main__':
    #plaser refactor accoridnt to the path
    # args will be better
    label_root_dir = '/var/datasets/rain_filtering/particle_labels/train'

    save_root_dir = '/var/datasets/rain_filtering/ann_dir/train'
    os.makedirs(save_root_dir, exist_ok=True)

    #sensor type
    label_list = ['first', 'last']
    
    image_list= np.sort(os.listdir(os.path.join(label_root_dir, label_list[0])))

    create_mmseg_labels(label_root_dir, label_list, image_list, save_root_dir)


