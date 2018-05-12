import os.path as path
import time

from immas.io import read_dataset
from immas.features.get_features import get_dataset_features_parallel

path_immas = "/home/okozyn/Projects/immas/"
path_dataset = path.join(path_immas, "dataset")

data_set = read_dataset(image_folder=path.join(path_dataset, "images"),
            mask_folder=path.join(path_dataset, "masks"),
            results_folder=path.join(path_dataset, "groundtruth"), 
            train_set_fraction=1)

print("Number of images for training is {0}, number of images for testing is {1}".format(
    len(data_set["train"]), len(data_set["test"])))

print("Started train feature extraction.")
features = get_dataset_features_parallel(data_set["train"], contour_max_number=None)
#print("Started test feature extraction.")
#features_test = get_dataset_features_parallel(data_set["test"], contour_max_number=None, train=False)

# this will delete all rows whith NaN values in it and save our data to a separate file
features = features.dropna()
#features_test = features_test.dropna()

#timestamp for file name
time_stamp = int(time.time())
file_name_train = f"examples/feature-tables/train-data_{time_stamp}.csv"
file_name_test = f"examples/feature-tables/test-data_{time_stamp}.csv"

features.to_csv(path.join(path_immas, file_name_train))
#features_test.to_csv(path.join(path_immas, file_name_test))
