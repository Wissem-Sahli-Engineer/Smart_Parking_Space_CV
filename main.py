import os 
import pickle
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from skimage.io import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

from sklearn.metrics import classification_report , accuracy_score

# DATA 

data = []
labels = []
groups = []

categories = ['empty', 'not_empty']
input = "data"

for cat_idx , cat in enumerate(categories):
    for file in os.listdir(os.path.join(input,cat)):
        img_path = os.path.join(input , cat , file)

        img = resize( imread(img_path) , (15,15))

        data.append(img.flatten())
        labels.append(cat_idx)
        groups.append(file.split('_')[1].split('.')[0])

data = np.asarray(data)
labels = np.asarray(labels)
groups = np.asarray(groups)

print(len(data),len(labels))

# Train / Test split grouped by Spot ID (suffix) to avoid spatial data leakage

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(data, labels, groups))

x_train = data[train_idx]
x_test = data[test_idx]
y_train = labels[train_idx]
y_test = labels[test_idx]


svm = SVC(C=10, gamma=0.01)

svm.fit(x_train, y_train)

y_pred_svm = svm.predict(x_train)

print("--- Résultats SVM ---")
print(classification_report(y_train, y_pred_svm))

score = accuracy_score ( svm.predict(x_test) , y_test )

print(score)

xgb = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    learning_rate = 0.05,

    # device = "cuda",
    tree_method = "hist",

    subsample = 0.8,
    colsample_bytree = 0.8,

    random_state = 42,
    eval_metric = 'logloss'
)

xgb.fit(x_train, y_train)

y_pred_xgb = xgb.predict(x_train)

print("--- XGBoost Results (Train) ---")
print(classification_report(y_train, y_pred_xgb))

rfc = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,
                            oob_score=True, n_jobs=-1)

rfc.fit(x_train,y_train)

y_rf_pred = rfc.predict(x_train)

print("--- RANDOM FOREST Results (Train) ---")
print(classification_report(y_train, y_rf_pred))

oob_accuracy = rfc.oob_score_
print(f"OOB Score (Global Precision) : {oob_accuracy:.4f}")


y_pred_xgb = xgb.predict(x_test)
print("---- XGBoost ----")
print(classification_report(y_test,y_pred_xgb))
print("")

y_pred_rf = rfc.predict(x_test)
print("---- RandomForest ----")
print(classification_report(y_test,y_pred_rf))
print("")

pickle.dump(svm , open("./model.p" , "wb"))