from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Convolution1D
from keras.initializers import random_uniform
import numpy as np
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
import ModelTemplates as MT
from keras.callbacks import History 
history = History()
from scipy.stats import pearsonr
from tqdm import tqdm
import pandas as pd
from keras.callbacks import *
import DNN_Plotter
from copy import deepcopy
import random
import keras

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File.npy") 
#parser.add_argument('-o', '--output', type=str, required=True, help = "output file path to save info about the train")
args = parser.parse_args()

def to_xy(df, target):
    y = df[:,target]
    x = np.delete(df, target, 1)
    return x,y

def mkdir_p(mypath):
    #crea una directory

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: 
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
    
def model_score(y_test, pred, th):
    print(">>>>Computing score th = {}".format(th))
    i=0
    while i<len(pred):
        if pred[i]>th:
            pred[i]=1;
        else:
            pred[i]=0;
        i=i+1
    return metrics.accuracy_score(y_test,pred)

def plot_conf_matrix(oos_test, pred, th, output_dir1):
    print(">>>>Confusion matrix th = {}".format(th))
    i = 0 
    while i < len(pred):
        if pred[i]>th:
            pred[i]=1
        else:
            pred[i]=0
        i=i+1
                            
    Confusion_matrix(output_dir1 + "/ConfusionMatrix",oos_test, pred, th)
    plt.clf()
    plt.close()
    
def plot_roc_curve(y_test, pred):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',
    label='0 pred power', alpha=.8)
    fp , tp, th = roc_curve(y_test, pred)
    roc = roc_auc_score(y_test, pred)
    plt.plot(fp, tp, 'r', label='ROC binary categorizzation (AUC = %0.3f)' %(roc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")
    
def plot_confusion_matrix(cm, names, title, cmap=plt.cm.Blues):
    #soglia per contrasto colore
    thresh = cm.max() / 1.5 
    #plotto per ogni cella della matrice il valore corrispondente
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black", fontsize=25)
    #plotto il resto
    #mp.rc('figure', figsize=(20,20), dpi=140)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, fontsize = 20)
    plt.yticks(tick_marks, names, fontsize = 20)
    plt.tight_layout()
    plt.ylabel('True label', fontsize = 20)
    plt.xlabel('Predicted label',fontsize = 20)
    plt.tight_layout()
    
    
def Confusion_matrix(output_dir1,oos_y, oos_pred, th):
    plt.rc('figure', figsize=(10,10), dpi=140)
    cm = confusion_matrix(oos_y, oos_pred)
    fig = plt.figure()
    #plt.rc('figure', figsize=(10,10), dpi=140)
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm,[0,1], title="Confusion matrix th {0}".format(th))
    #plt.draw()
    fig.savefig(output_dir1 + "/Not Normalized Confusion Matrix {}.png".format(th))
    plt.clf()
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig1 = plt.figure()
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix th {}'.format(th))
    #plt.draw()
    fig1.savefig(output_dir1 + "/normkfold_{0}.png".format(th))
    plt.clf()
    plt.cla()
    plt.close()
    

def Model(input_dim):
    print(">>> Creating model...")
    model = Sequential()
    model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
    model.add(Dense(units=400,activation="relu"))
    model.add(Dense(units=300,activation="relu"))
    model.add(Dense(units=100,activation="relu"))
    model.add(Dense(units=5,activation="relu"))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[keras.metrics.binary_accuracy])
    return model

data = np.load(args.file)
print(data.shape)
x,y = to_xy(data, data.shape[1]-1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
output_dir = "/home/giacomo/DNN_test"
mkdir_p(output_dir)
print(">>> Testing model...")
early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1, mode='auto', baseline=None)
plt.rc('figure', figsize=(15,10), dpi=140)
csv_logger = CSVLogger(output_dir +'/training.log')
model = Model(x_train.shape[1])
model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = 1000, batch_size = 1000, class_weight=class_weights,shuffle=True,
                            callbacks=[csv_logger, early_stop])
pred = model.predict(x_test)

#>>>>>>Plotting<<<<<<<

th = 0.3
i=0
th = 0.3
score = []

plot_roc_curve(y_test, pred)

mkdir_p(output_dir + "/ConfusionMatrix")
th = 0.3
while th < 1:
    prova = deepcopy(pred)
    plot_conf_matrix(y_test, prova, th, output_dir)
    th = th + 0.1
    th = round(th, 1)
                            
while th < 1:
    prova = deepcopy(pred)
    z = model_score(y_test, prova, th)
    score.append([th, z])
    th = th + 0.1
    th = round(th, 1)
    
"""
 print(">>> Saving Loss and Val Loss...")
mkdir_p(output_dir1 + "/Loss_models")
DNN_Plotter.Loss_val_loss_prova("loss", "loss model", output_dir + "/Loss_models")
plt.clf()
plt.close()
mkdir_p(output_dir1 + "/Val_loss_model")
DNN_Plotter.Loss_val_loss_prova("val_loss", "val_loss model", output_dir + "/Val_loss_model")
plt.clf()
plt.close()
"""
