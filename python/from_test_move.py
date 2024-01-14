import os
import shutil

for f in os.listdir('test'):
    if os.path.exists("index/" + f) and os.path.exists("F:/Kursinis/train_moved/" + f):
        shutil.copytree("index/" + f, "index_split/" + f);
        shutil.copytree("F:/Kursinis/train_moved/" + f, "train_split/" + f);
