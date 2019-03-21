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
    
def plot_roc_curve_KFold(tprs, mean_fpr, aucs):
    #plotto la bisettrice, potere di predizione nullo
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='0 pred power', alpha=.8)
    #valore medio del true positive rate
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    #valore medio auc

    #mean_auc = auc(mean_fpr, mean_tpr)
    #std deviation auc
    #std_auc = np.std(aucs)
    
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC ',lw=2, alpha=.8) #(AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
    #deviazione standard sui true positive rate (ordinate) prendo il valore medio e aggiungo o 
    #sottraggo la dev standard per riempire l'area di incertezza dell roc curve media sui kfolds
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1,
                 label=r'$\pm$ 1 std. dev.')
    #plotto labels, titolo, dimensiono assi e legenda
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")

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


def Confusion_matrix(output_dir1,oos_y, oos_pred, models, th):
    plt.rc('figure', figsize=(10,10), dpi=140)
    cm = confusion_matrix(oos_y, oos_pred)
    fig = plt.figure()
    #plt.rc('figure', figsize=(10,10), dpi=140)
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm,[0,1], title="Confusion matrix modello {0} th {1}".format(models, th))
    #plt.draw()
    fig.savefig(output_dir1 + "/nonnormkfold_{}.png".format(th))
    plt.clf()
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig1 = plt.figure()
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix {}'.format(th))
    #plt.draw()
    fig1.savefig(output_dir1 + "/normkfold_{0}.png".format(th))
    plt.clf()
    plt.cla()
    plt.close()
    
def Confusion_matrix_best(output_dir1,oos_y, oos_pred, th):
    plt.rc('figure', figsize=(10,10), dpi=140)
    cm = confusion_matrix(oos_y, oos_pred)
    fig = plt.figure()
    #plt.rc('figure', figsize=(10,10), dpi=140)
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm,[0,1], title="Confusion matrix modello th {0}".format( th))
    #plt.draw()
    fig.savefig(output_dir1 + "/nonnormkfold_{}.png".format(th))
    plt.clf()
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig1 = plt.figure()
    #plt.figure(figsize=(10,8))
    plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix {}'.format(th))
    #plt.draw()
    fig1.savefig(output_dir1 + "/normkfold_{0}.png".format(th))
    plt.clf()
    plt.cla()
    plt.close()
    
def Loss_val_loss(modelli, lab, legend, outdir,k):
    fold = 1
    for models in modelli:
        while fold < k+1:
            d = pd.read_csv(outdir + "/training_{0}.log".format(fold))
            plt.plot(d[lab], label = legend + "{0} fold # {1}".format(models, fold))
            fold += 1
    
        figure = plt.gcf()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(lab)
        figure.savefig(outdir+"/model_{0}_{1}".format(models,lab))
        
def Loss_val_loss_prova(modelli, lab, legend, outdir,k, f, j):
    fold = 1
    k = int(k)
    f = int(f)
    for models in modelli:
        while fold < k+1:
            d = pd.read_csv("/home/giacomo/tesi1/DNN_test/third dataset/DNNoptimizer/megadataset/Kfold_modello_{0}_{1}_{2}/training_{3}.log".format(models,f,j,fold))
            plt.plot(d[lab], label = legend + "{0}_{1}_{2} fold # {3}".format(models,f, j, fold))
            fold += 1
    
        figure = plt.gcf()
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel(lab)
        figure.savefig(outdir+"/model_{0}_{1}_{2}.png".format(models,f,j))

