import os
import shutil
import random
from tqdm import tqdm


def count_files_in_directory(directory):
    try:
        files = os.listdir(target_directory + "/" + directory)
        file_count = len(files)
        return file_count
    except Exception as e:
        print(f"Error counting files in directory {directory}: {e}")
        return -1


def check_directory(directory):
    print("Checking directory: " + directory)
    files = os.listdir(directory)
    random.shuffle(files)
    for dir in tqdm(files):
        file_count = count_files_in_directory(dir)
        if file_count > 500:
            os.mkdir("train_split_500/" + dir)
            imgs = os.listdir(target_directory + "/" + dir)
            random.shuffle(imgs)
            for i, file in enumerate(imgs):
                if i < 500:
                    shutil.copy(
                        target_directory + "/" + dir + "/" + file,
                        "train_split_500/" + dir + "/" + file,
                    )


target_directory = "/mnt/f/Bakis/train_moved"
check_directory(target_directory)
