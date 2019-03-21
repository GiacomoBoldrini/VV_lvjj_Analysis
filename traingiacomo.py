"""traingiacomo esegue due tipi di training su un dataset creato con JetSelectionDatasetCreator.py. Possiamo scegliere un training "semplice" oppure un training fatto su KFold. Entrambi stampano in formato .png la roc curve e un file .txt con 
informazioni base del training. Possiamo anche scegliere di stampare una confusion matrix e impostare la soglia per il confronto dei parametri (di base sarà 0.5 e con rint approssimiamo all'intero più vicino).
alcune linee da comando utili:

-Training semplice senza CM stampa solo file .txt e roc curve in .png:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10

-Training semplice con CM con soglia a 0.7, stampa file .txt, roc curve in .png, confusion matrix normalizzata e non:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -m -th 0.7

-Training KFold 5 fold senza CM stampa solo file .txt e roc curve in .png:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -i 5

-Training KFold 5 fold con CM con soglia a 0.7, stampa file .txt, roc curve in .png, confusion matrix normalizzata e non:
python traingiacomo.py -f dataset.npy -o /path/per/il/.txt/e/.png -e 10 -m -th 0.7 -i 5

"""

import ROOT as r
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mp
import itertools
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import sys
from array import array
from VBSAnalysis.EventIterator import EventIterator
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

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    #soglia per contrasto colore
    thresh = cm.max() / 1.5 
    #plotto per ogni cella della matrice il valore corrispondente
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black", fontsize=25)
    #plotto il resto
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
    mean_auc = auc(mean_fpr, mean_tpr)
    #std deviation auc
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
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

if args.crossval :
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    kf = KFold(args.crossval)
    oos_y = []
    oos_pred = []
    fold = 0
    tps = []
    aucs = []
    mean_fp = np.linspace(0, 1, 100)
    for train,test in kf.split(x):
        fold+=1
        print("Fold #{}".format(fold))
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        model = Sequential()
        model.add(Dense(50, input_dim = x.shape[1], activation = 'relu' ))
        model.add(Dense(30, activation = 'relu'))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(1, activation= 'sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = epoch, batch_size=32, class_weight=class_weights)
        predi = model.predict(x_test)
        oos_y.append(y_test)
        oos_pred.append(predi)
        fp , tp, th = roc_curve(y_test, predi)
        tps.append(interp(mean_fp, fp, tp))
        tps[-1][0] = 0.0
        roc_auc = roc_auc_score(y_test, predi)
        aucs.append(roc_auc)
        plt.plot(fp, tp, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.3f)' % (fold, roc_auc))  
    model.save("modelli_giacomo/PrimoKFold5")    
    mp.rc('figure', figsize=(10,8), dpi=140)
    fig0 = plt.gcf()
    plot_roc_curve_KFold(tps, mean_fp, aucs)
    plt.figure(figsize=(20,20))
    plt.draw()
    fig0.savefig(args.output + "/roc_PJAssoc_kfold.png")
    plt.clf()
    plt.cla()
    plt.close()
    oos_y = np.concatenate(oos_y)
    oos_pred = np.concatenate(oos_pred)
    score = model_score(oos_y, oos_pred)
    auc = roc_auc_score(oos_y, oos_pred)
    if args.confmatrix:
        mp.rc('figure', figsize=(12,8), dpi=140)
        if not args.threshold:
            oos_pred = np.rint(oos_pred)
        else:
            th = args.threshold
            i=0
            while i<len(oos_pred):
                if oos_pred[i]>th:
                    oos_pred[i]=1;
                else:
                    oos_pred[i]=0;
                i=i+1
        cm = confusion_matrix(oos_y, oos_pred)
        fig = plt.gcf()
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cm,[0,1], title='Not normalized confusion matrix')
        plt.draw()
        fig.savefig(args.output + "/nonnormkfold.png")
        plt.clf()
        plt.cla()
        plt.close()
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig1 = plt.gcf()
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix')
        plt.draw()
        fig1.savefig(args.output + "/normkfold.png")
        plt.clf()
        plt.cla()
        plt.close()

    if not args.evaluate:
        print(">>> Saving parameters...")
        f = open(args.output + "/configsKFold.txt", "w")
        f.write("epochs: {0}\n".format(epoch))
        f.write("folds: {0}\n".format(args.crossval))
        #f.write("model_schema: {0}\n".format(args.model_schema))
        f.write("batch_size: {0}\n".format(32))
        f.write("AUC: {0}\n".format(auc))
        f.close()
       
       
else:
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    model = Sequential()
    model.add(Dense(50, input_dim = x.shape[1], activation = 'relu' ))
    model.add(Dense(30, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, activation= 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x,y,validation_data=(x_test,y_test),verbose=1,epochs = epoch, batch_size=32, class_weight=class_weights)
    model.save("modelli_giacomo/modellobinariojp")
    pred = model.predict(x_test)
    score = model_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    mp.rc('figure', figsize=(11,8), dpi=140)
    fig0 = plt.gcf()
    plot_roc_curve(y_test, pred)
    plt.figure(figsize=(20,20))
    plt.draw()
    fig0.savefig(args.output + "/roc_PJAssoc_bin.png")
    plt.clf()
    if args.confmatrix:
        mp.rc('figure', figsize=(10,8), dpi=140)
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
        cm = confusion_matrix(y_test, pred)
        plot_confusion_matrix(cm,[0,1], title='Not normalized confusion matrix')
        plt.draw()
        plt.savefig(args.output + "/no_norm_no_cross.png")
        plt.clf()
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix')
        plt.draw()
        plt.savefig(args.output + "/norm_no_cross.png")
    if not args.evaluate:
        print(">>> Saving parameters...")
        f = open(args.output + "/configssimple.txt", "w")
        f.write("epochs: {0}\n".format(epoch))
        #f.write("model_schema: {0}\n".format(args.model_schema))
        f.write("batch_size: {0}\n".format(32))
        f.write("AUC: {0}\n".format(auc))
        f.close()
