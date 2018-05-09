import os.path as path

from immas.io import read_dataset
from immas.features.get_features import get_dataset_features_parallel

path_immas = "/home/okozyn/Projects/immas/"
path_dataset = "/home/okozyn/Projects/immas/dataset/"

data_set = read_dataset(image_folder=path.join(path_dataset, "masses_examples"),
            mask_folder=path.join(path_dataset, "masks"),
            results_folder=path.join(path_dataset, "groundtruth"),
            pmuscle_mask_folder=path.join(path_dataset, "pectoral_muscle_masks"), train_set_fraction=1)

print("Number of images for training is {0}, number of images for testing is {1}".format(
    len(data_set["train"]), len(data_set["test"])))

imgs = data_set["train"]

print("Started feature extraction.")
features = get_dataset_features_parallel(imgs, contour_max_number=None)

# this will delete all rows whith NaN values in it and save our data to a separate file
features = features.dropna()
features.to_csv(path.join(path_immas, "/debug/classifier-train-data.csv"))