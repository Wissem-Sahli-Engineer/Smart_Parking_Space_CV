import os 
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from skimage.io import imread
from skimage.transform import resize

data = []
labels = []

categories = ['empty', 'not_empty']
input = "data"

for cat_idx , cat in enumerate(categories):
    for file in os.listdir(os.path.join(input,cat)):
        img_path = os.path.join(input , cat , file)

        img = resize( imread(img_path) , (15,15))

        data.append(img.flatten())
        labels.append(cat_idx)

data = np.asarray(data)
labels = np.asarray(labels)

print(len(data),len(labels))