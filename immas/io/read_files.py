import os, sys

def get_images_list(path='./dataset', file_extentions=[".tiff"]):
    '''
    Function for finding all images in the specified folder.

    Args:
        path (str): path to the folder
        file_extentions ([str]): list of file extentions to include

    Returns:
        ([str], [str]): list of file names, list of the corresponding paths    
    '''

    file_names = []
    file_paths = []

    # we should have different processing for different projects in data set

    for dir_name, subdir_list, file_list in os.walk(path):
        for file_name in file_list:
            for file_ext in file_extentions:
                if file_ext in file_name.lower():
                    file_names.append(file_name)
                    file_paths.append(os.path.join(dir_name, file_name))

    return file_names, file_paths                
