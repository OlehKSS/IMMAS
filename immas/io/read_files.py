import os, sys
from random import shuffle
from math import floor
from .mammogram import MammogramImage

def read_dataset(image_folder, mask_folder, results_folder, pmuscle_mask_folder, 
                 train_set_fraction=0.25):
    '''
    Reads dataset and returns list of mammogram images found.
        
    Args:
        image_folder (str): path to the images.
        mask_folder (str): path to the mask for the images.        
        results_folder (str): path to the corectly segmented images.
        pmuscle_mask_folder (str): path to the pectoral muscle masks.
        train_set_fraction (float): fraction of the data to be used for training. 
        Default is 25%.

    Returns:
        {"train": [MammogramImage], "test": [MammogramImage]}: dictionary 
        of lists with training and test mammogram images found. 
    '''

    mask_extenstions = [".png"]
    imgs_mass = []
    imgs_clean = []

    print("Reading list of files...")
    
    images = get_images(image_folder)
    masks = get_images(mask_folder, mask_extenstions)
    results = get_images(results_folder)
    pmuscle_mask = get_images(pmuscle_mask_folder)

    if not len(images):
        raise RuntimeError("Could not find any image files.")

    if not len(masks):
        raise RuntimeError("Could not find any image mask files.")

    print("Reading mamograms images and all additional data...")

    # checking whether we have groundtruth (mass) or not in order to divide the dataset
    # into images with masses and without ones 
    for exam_name in images:
        temp_new_mm_img = MammogramImage(image_path=images[exam_name],
                                        mask_path=masks[exam_name],
                                        ground_truth_path=results.get(exam_name),
                                        pmuscle_mask_path=pmuscle_mask.get(exam_name),
                                        load_data=False,
                                        file_name=exam_name)
        if results.get(exam_name):
            imgs_mass.append(temp_new_mm_img)
        else:
            imgs_clean.append(temp_new_mm_img)   
    
    # random partitioning and shuffling of the data into train and test dataset
    train_imgs_mass_len = floor(train_set_fraction*len(imgs_mass)) + 1
    train_imgs_clean_len = floor(train_set_fraction*len(imgs_clean)) + 1

    shuffle(imgs_mass)
    shuffle(imgs_clean)

    train_imgs = imgs_mass[1:train_imgs_mass_len] + imgs_clean[1:train_imgs_clean_len]
    test_imgs = imgs_mass[train_imgs_mass_len:] + imgs_clean[train_imgs_clean_len:]

    shuffle(train_imgs)
    shuffle(test_imgs)

    mammogram_images = {"train": train_imgs, "test": test_imgs}

    print("All data have been successfully loaded.")

    return mammogram_images    


def get_images(path="./dataset", file_extentions=[".tif"]):
    '''
    Function for finding all images in the specified folder.

    Args:
        path (str): path to the folder
        file_extentions ([str]): list of file extentions to include

    Returns:
        dict(str, str): dictionary of file names and corresponding paths    
    '''

    # dictionary instantiation
    file_names_and_path = {}

    # we should have different processing for different projects in data set

    for dir_name, subdir_list, file_list in os.walk(path):
        for file_name in file_list:
            for file_ext in file_extentions:
                if file_ext in file_name.lower():
                    # save filename without extenstion (four last characters)
                    file_names_and_path[file_name[:-4]] = os.path.join(dir_name, file_name)

    return file_names_and_path                

