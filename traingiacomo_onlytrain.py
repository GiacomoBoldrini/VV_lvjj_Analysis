"""

-Training semplice senza CM stampa solo file .txt e roc curve in .png:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10

-Training semplice con CM con soglia a 0.7, stampa file .txt, roc curve in .png, confusion matrix normalizzata e non:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -m -th 0.7

-Training KFold 5 fold senza CM stampa solo file .txt e roc curve in .png:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -i 5

-Training KFold 5 fold con CM con soglia a 0.7, stampa file .txt, roc curve in .png, confusion matrix normalizzata e non:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -m -th 0.7 -i 5

"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
import sys
from array import array
import argparse
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import KFold
from scipy import interp

def to_xy(df, target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y

    
def model_score(y_test, pred):
    if not args.threshold:
        pred = np.rint(pred)
    else:
        th = args.threshold
        i=0
        while i<len(pred):
            if pred[i]>th:
                pred[i]=1;
            else:
                pred[i]=0;
            i=i+1
    return metrics.accuracy_score(y_test,pred)

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File.npy") 
parser.add_argument('-i', '--crossval', type=int, required=False, help="K-Fold cross validation number") 
parser.add_argument('-m', '--confmatrix',  required=False, help="If we want to plot confusion matrix with threshold at 0.5 for jet 0 or jet 1 ", action = "store_true")
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-o', '--output', type=str, required=True, help = "output file path to save info about the train")
parser.add_argument('-e', '--epochs', type=int, required=False, help = "number of epochs for fit model")
parser.add_argument('-th', '--threshold', type=float, required=False, help = "threshold for computing consufion matrix and scores. default 0.5")
args = parser.parse_args()

if not args.epochs:
    epoch = 10
else:
    epoch = args.epochs
    
data = np.load(args.file)
x,y = to_xy(data, 10)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
model = Sequential()
model.add(Dense(100, input_dim = x.shape[1], activation = 'relu' ))
model.add(Dropout(0.2))
model.add(Dense(80, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation= 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x,y,validation_data=(x_test,y_test),verbose=1,epochs = epoch, batch_size=1024, class_weight=class_weights)
model.save("modelli_giacomo/modellobinariojp")
pred = model.predict(x_test)
score = model_score(y_test, pred)
auc = roc_auc_score(y_test, pred)
print("AUC:", auc)

if not args.evaluate:
    print(">>> Saving parameters...")
    f = open(args.output + "/configssimple.txt", "w")
    f.write("epochs: {0}\n".format(epoch))
    #f.write("model_schema: {0}\n".format(args.model_schema))
    f.write("batch_size: {0}\n".format(32))
    f.write("AUC: {0}\n".format(auc))
    f.close()
