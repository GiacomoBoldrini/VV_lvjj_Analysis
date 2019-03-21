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

def plot_conf_matrix(oos_test, pred, th, output_dir1, models):
    print(">>>>Confusion matrix th = {}".format(th))
    i = 0 
    while i < len(pred):
        if pred[i]>th:
            pred[i]=1
        else:
            pred[i]=0
        i=i+1
                            
    DNN_Plotter.Confusion_matrix(output_dir1 + "/ConfusionMatrix",oos_test, pred,models, th)
    plt.clf()
    plt.close()
    
    
    
    
    
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help="File.npy") 
parser.add_argument('-i', '--crossval', type=int, required=False, help="K-Fold cross validation number") 
parser.add_argument('-cm', '--confmatrix',  required=False, help="If we want to plot confusion matrix with threshold at 0.5 for jet 0 or jet 1 ", action = "store_true")
parser.add_argument('-ev', '--evaluate', action="store_true")
parser.add_argument('-e', '--epochs', type=int, required=False, help = "number of epochs for fit model")
parser.add_argument('-bs', '--batch', type=int, required=True, help = "batch size (1024 or 32)")

args = parser.parse_args()

if not args.epochs:
    epoch = 10
else:
    epoch = args.epochs
    
data = np.load(args.file)
print(data.shape)
random.shuffle(data)
x,y = to_xy(data, 17)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)

output_dir = "/home/giacomo/tesi1/DNN_test/Convolutional"
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
kf = KFold(args.crossval)
output_dir1 = output_dir + "/Kfold_convolutional"
oos_y = []
oos_pred = []
fold = 0
tps = []
aucs = []
mean_fp = 0
mean_fp = np.linspace(0, 1, 100)
print(">>> Testing model K fold ({0})...".format("convolution"))
mkdir_p(output_dir1)
early_stop = EarlyStopping(monitor='loss', min_delta=1e-15, patience=15, verbose=1, mode='auto', baseline=None)                    
for train,test in kf.split(x):
    plt.rc('figure', figsize=(15,10), dpi=140)
    fold+=1
    print(" Fold #{}".format(fold))
    csv_logger = CSVLogger(output_dir1 +'/training_{0}.log'.format(fold))
    x_train = x[train]
    #expanding dimension such that x_train was (a,b) now (a,b,1) for convolution
    x_train = np.expand_dims(x_train, axis=2)
    print(x_train.shape)
    y_train = y[train]
    print(y_train.shape)
    x_test = x[test]
    x_test = np.expand_dims(x_test, axis=2)
    print(x_test.shape)
    y_test = y[test]
    print(y_test.shape)
    model = MT.Convolutional(x_train.shape)
    model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = epoch, batch_size=args.batch, class_weight=class_weights,shuffle=True,
    callbacks=[csv_logger, early_stop])
    predi = model.predict(x_test)
    oos_y.append(y_test)
    oos_pred.append(predi)
    fp , tp, th = roc_curve(y_test, predi)
    tps.append(interp(mean_fp, fp, tp))
    tps[-1][0] = 0.0
    roc_auc = roc_auc_score(y_test, predi)
    aucs.append(roc_auc)
    plt.plot(fp, tp, lw=1, alpha=0.3,
    label='{} ROC fold %d (AUC = %0.3f)'.format("convolutional") % (fold, roc_auc))                
model.save(output_dir1 + "/modello_{}".format("convolutional")  )  
fig0 = plt.gcf()
DNN_Plotter.plot_roc_curve_KFold(tps, mean_fp, aucs)
fig0.savefig(output_dir1 + "/roc_modello_{}".format("Convolutional"))
plt.clf()
plt.close()
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
oos_pred = np.concatenate(oos_pred)
print(oos_y)
print(oos_pred)
                    
if args.confmatrix:
    mkdir_p(output_dir1 + "/ConfusionMatrix")
    th = 0.1
    while th < 1:
        prova = deepcopy(oos_pred)
        plot_conf_matrix(oos_y[:], prova, th, output_dir1, "Convolutional")
        th = th + 0.1
        th = round(th, 1)
                            
th = 0.1
score = []
while th < 1:
    prova = deepcopy(oos_pred)
    z = model_score(oos_y, prova, th)
    score.append([th, z])
    th = th + 0.1
    th = round(th, 1)
    
auc = roc_auc_score(oos_y, oos_pred)
                    
if not args.evaluate:
    print(">>> Saving parameters...")
    f = open(output_dir1 + "/configsKFold.txt", "w")
    f.write("epochs: {0}\n".format(epoch))
    f.write("folds: {0}\n".format(args.crossval))
    f.write("batch_size: {0}\n".format(args.batch))
    f.write("AUC: {0}\n".format(auc))
    i = 0
    while i < len(score):
        f.write("Score: {0}\n".format(score[i]))
        i += 1
    f.close()
        
"""
print(">>> Saving Loss and Val Loss...")
mkdir_p(output_dir1 + "/Loss_models")
DNN_Plotter.Loss_val_loss_prova(args.model, "loss", "loss model", output_dir1 + "/Loss_models", args.crossval, first, second)
plt.clf()
plt.close()
mkdir_p(output_dir1 + "/Val_loss_model")
DNN_Plotter.Loss_val_loss_prova(args.model, "val_loss", "val_loss model", output_dir1 + "/Val_loss_model", args.crossval, first, second)
plt.clf()
plt.close()
"""
        
