import os
import shutil
import random
from tqdm import tqdm

def count_files_in_directory(directory):
    try:
        # List all files in the directory
        files = os.listdir(target_directory + "/" + directory)
        # Count the number of files
        file_count = len(files)
        return file_count
    except Exception as e:
        print(f"Error counting files in directory {directory}: {e}")
        return -1


def check_directory(directory):
    n = 0;
    print("Checking directory: " + directory)
    files = os.listdir(directory);
    random.shuffle(files)
    for dir in tqdm(files):
        if n >= 5000:
            break
        file_count = count_files_in_directory(dir)
        if file_count > 25:
            os.mkdir("train_split_25/" + dir)
            for i, file in enumerate(os.listdir(target_directory + "/" + dir)):
                if i < 25:
                    shutil.copy(target_directory + "/" + dir + "/" + file, 
                            "train_split_25/" + dir + "/" + file)
            n += 1

target_directory = "F:/Kursinis/train_moved"
check_directory(target_directory)
