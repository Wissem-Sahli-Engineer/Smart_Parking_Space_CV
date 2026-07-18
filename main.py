import os 
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifiers
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

# DATA 

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

# Train / Test split

x_train , x_train , y_train , y_test = train_test_split( data , labels , 
                                                test_size = 0.2, 
                                                shuffle = True,
                                                stratify= labels,
                                                random_state=42)




