""" from command line example:
python traingiacomo.py -m binary_small binary_dropout binary_dropout_small -f dataset.npy -om /output/path/to/save/model -e 1000 

-m is a list of model's id so we can iterate in a for loop and save many models. 
The model names can be found in ModelTemplate.py"""

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
parser.add_argument('-m', '--model', nargs='+', required=True)
parser.add_argument('-f', '--file', type=str, required=True, help="File.npy") 
parser.add_argument('-i', '--crossval', type=int, required=False, help="K-Fold cross validation number") 
parser.add_argument('-cm', '--confmatrix',  required=False, help="If we want to plot confusion matrix with threshold at 0.5 for jet 0 or jet 1 ", action = "store_true")
parser.add_argument('-ev', '--evaluate', action="store_true")
#parser.add_argument('-o', '--output', type=str, required=True, help = "output file path to save info about the train")
parser.add_argument('-e', '--epochs', type=int, required=False, help = "number of epochs for fit model")
parser.add_argument('-bs', '--batch', type=int, required=True, help = "batch size (1024 or 32)")
parser.add_argument('-th', '--threshold', type=float, required=False, help = "threshold for computing consufion matrix and scores. default 0.5")
parser.add_argument('-d', '--dense', nargs='+', required=True)

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

if not args.dense:
    if args.crossval :
        output_dir = "/home/giacomo/tesi1/DNN_test/third dataset/DNNoptimizer"
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
        kf = KFold(args.crossval)
        for models in args.model:
            output_dir1 = output_dir + "/Kfold_modello_{}".format(models)
            oos_y = []
            oos_pred = []
            fold = 0
            tps = []
            aucs = []
            mean_fp = 0
            mean_fp = np.linspace(0, 1, 100)
            print(">>> Testing model K fold ({0})...".format(models))
            mkdir_p(output_dir1)
            early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10, verbose=1, mode='auto', baseline=None)
            
            for train,test in kf.split(x):
                plt.rc('figure', figsize=(15,10), dpi=140)
                fold+=1
                print(" Fold #{}".format(fold))
                csv_logger = CSVLogger(output_dir1 +'/training_{0}.log'.format(fold))
                x_train = x[train]
                y_train = y[train]
                x_test = x[test]
                y_test = y[test]
                model = MT.ModelTemplates(models, x_train.shape[1])
                model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = epoch, batch_size=args.batch, class_weight=class_weights,shuffle=True,
                            callbacks=[csv_logger, early_stop])
                predi = model.predict(x_test)
                oos_y.append(y_test)
                count0 = 0
                count1 = 0
                for i in y_test:
                    print(i)
                    if i == 0:
                        count0 += 1
                    else:
                        count1 += 1
                print('count0')
                print(count0)
                print('count1')
                print(count1)
                oos_pred.append(predi)
                fp , tp, th = roc_curve(y_test, predi)
                tps.append(interp(mean_fp, fp, tp))
                tps[-1][0] = 0.0
                roc_auc = roc_auc_score(y_test, predi)
                aucs.append(roc_auc)
                plt.plot(fp, tp, lw=1, alpha=0.3,
                    label='{} ROC fold %d (AUC = %0.3f)'.format(models) % (fold, roc_auc))  

            model.save(output_dir1 + "/modello_{}".format(models))    
            #mp.rc('figure', figsize=(10,8), dpi=140)
            fig0 = plt.gcf()
            #plt.rc('figure', figsize=(15,10), dpi=140)
            DNN_Plotter.plot_roc_curve_KFold(tps, mean_fp, aucs)
            #plt.figure(figsize=(20,20))
            #plt.draw()
            fig0.savefig(output_dir1 + "/roc_modello_{}".format(models))
            plt.clf()
            plt.close()
            oos_y = np.concatenate(oos_y)
            oos_pred = np.concatenate(oos_pred)
            oos_pred = np.concatenate(oos_pred)
            auc = roc_auc_score(oos_y, oos_pred)
            score = model_score(oos_y, oos_pred)
            if args.confmatrix:
                mkdir_p(output_dir1 + "/ConfusionMatrix")
                th = 0.3
                i=0
                while th < 1:
                    i=0
                    while i<len(oos_pred):
                        if oos_pred[i]>th:
                            oos_pred[i]=1;
                        else:
                            oos_pred[i]=0;
                        i=i+1
                    
                    DNN_Plotter.Confusion_matrix(output_dir1 + "/ConfusionMatrix",oos_y, oos_pred,models, th)
                    plt.clf()
                    plt.close()
                    th = th + 0.1
                    th = round(th, 1)
                    
            score = model_score(oos_y, oos_pred)
                
            if not args.evaluate:
                print(">>> Saving parameters...")
                f = open(output_dir1 + "/configsKFold.txt", "w")
                f.write("model: {0}\n".format(models))
                f.write("epochs: {0}\n".format(epoch))
                f.write("folds: {0}\n".format(args.crossval))
                #f.write("model_schema: {0}\n".format(args.model_schema))
                f.write("batch_size: {0}\n".format(args.batch))
                f.write("AUC: {0}\n".format(auc))
                f.write("Score: {0}\n".format(score))
                
                f.close()
    
    #l'else non lo uso perchè uso solo il k-fold per avere tutto il dataset. Dalla konsole bisogna mettere il valore 
    #k-fold splitting (in genere metto 5)
    """
    else:
        output_dir = "/home/giacomo/tesi1/DNN_test"
        x_train, x_test, y_train, y_test = train_tes#score = model_score(y_test, pred)t_split(x,y,test_size=0.25,random_state=42)
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
        for models in args.model:
            mkdir_p(output_dir + "/modello_{}".format(models))
            output_dir1 = output_dir + "/modello_{}".format(models)
            print(">>> Testing model ({0})...".format(models))
            model = MT.ModelTemplates(models, x_train.shape[1])
            model.fit(x,y,validation_data=(x_test,y_test),verbose=1,epochs = epoch, batch_size=args.batch, class_weight=class_weights)
            model.save(output_dir1)
            pred = model.predict(x_test)
            score = model_score(y_test, pred)
            auc = roc_auc_score(y_test, pred)
            mp.rc('figure', figsize=(11,8), dpi=140)
            fig0 = plt.gcf()
            plot_roc_curve(y_test, pred)
            plt.figure(figsize=(20,20))
            plt.draw()
            fig0.savefig(output_dir1+ "/roc_model_{}.png".format(models))
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
                plot_confusion_matrix(cm,[0,1], title="Confusion matrix modello {}".format(models))
                plt.draw()
                plt.savefig(output_dir1 + "/no_norm_no_cross.png")
                plt.clf()
                np.set_printoptions(precision=2)
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix')
                plt.draw()
                plt.savefig(args.output1 + "/norm_no_cross.png")
            if not args.evaluate:
                print(">>> Saving parameters...")
                f = open(args.output + "/configssimple.txt", "w")
                f.write("model: {0}\n".format(models))
                f.write("epochs: {0}\n".format(epoch))
                #f.write("model_schema: {0}\n".format(args.model_schema))
                f.write("batch_size: {0}\n".format(args.batch))
                f.write("AUC: {0}\n".format(auc))
                f.close()
    """
    print(">>> Saving Loss and Val Loss...")
    DNN_Plotter.Loss_val_loss(args.model, "loss", "loss model", output_dir + "/Loss_models", args.crossval)
    plt.clf()
    plt.close()
    DNN_Plotter.Loss_val_loss(args.model, "val_loss", "val_loss model", output_dir + "/Val_loss_model", args.crossval)
    plt.clf()
    plt.close()
    
else: 
    denso = [10, 50, 75, 100, 150, 200, 300]
    
    for first in denso:
        for second in args.dense:
            
            if args.crossval :
                output_dir = "/home/giacomo/tesi1/DNN_test/third dataset/DNNoptimizer/megadataset"
                class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
                kf = KFold(args.crossval)
                for models in args.model:
                    output_dir1 = output_dir + "/Kfold_modello_{}_{}_{}".format(models,first,second)
                    oos_y = []
                    oos_pred = []
                    fold = 0
                    tps = []
                    aucs = []
                    mean_fp = 0
                    mean_fp = np.linspace(0, 1, 100)
                    print(">>> Testing model K fold ({0})...".format(models))
                    mkdir_p(output_dir1)
                    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-20, patience=20, verbose=1, mode='auto', baseline=None)
                    
                    for train,test in kf.split(x):
                        plt.rc('figure', figsize=(15,10), dpi=140)
                        fold+=1
                        print(" Fold #{}".format(fold))
                        csv_logger = CSVLogger(output_dir1 +'/training_{0}.log'.format(fold))
                        x_train = x[train]
                        y_train = y[train]
                        x_test = x[test]
                        y_test = y[test]
                        model = MT.ModelProva(models, x_train.shape[1], first, second)
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
                            label='{} ROC fold %d (AUC = %0.3f)'.format(models) % (fold, roc_auc))  

                    model.save(output_dir1 + "/modello_{}_{}_{}".format(models,first,second))    
                    #mp.rc('figure', figsize=(10,8), dpi=140)
                    fig0 = plt.gcf()
                    #plt.rc('figure', figsize=(15,10), dpi=140)
                    DNN_Plotter.plot_roc_curve_KFold(tps, mean_fp, aucs)
                    #plt.figure(figsize=(20,20))
                    #plt.draw()
                    fig0.savefig(output_dir1 + "/roc_modello_{}_{}_{}".format(models,first,second))
                    plt.clf()
                    plt.close()
                    oos_y = np.concatenate(oos_y)
                    oos_pred = np.concatenate(oos_pred)
                    oos_pred = np.concatenate(oos_pred)
                    print(oos_y)
                    print(oos_pred)
                    
                    if args.confmatrix:
                        mkdir_p(output_dir1 + "/ConfusionMatrix")
                        th = 0.3
                        while th < 1:
                            prova = deepcopy(oos_pred)
                            plot_conf_matrix(oos_y[:], prova, th, output_dir1, models)
                            th = th + 0.1
                            th = round(th, 1)
                            
                    th = 0.3
                    score = []
                    while th < 1:
                        prova = deepcopy(oos_pred)
                        z = model_score(oos_y, prova, th)
                        score.append([th, z])
                        th = th + 0.1
                        th = round(th, 1)
                    auc = roc_auc_score(oos_y, oos_pred)
                    """if args.confmatrix:
                        mkdir_p(output_dir1 + "/ConfusionMatrix")
                        th = 0.3
                        while th < 1:
                            print(oos_pred)
                            plot_conf_matrix(oos_y[:], oos_pred[:], th, output_dir1, models)
                            th = th + 0.1
                            th = round(th, 1)"""
                    if not args.evaluate:
                        print(">>> Saving parameters...")
                        f = open(output_dir1 + "/configsKFold.txt", "w")
                        f.write("first layer: {0}\n".format(first))
                        f.write("second layer: {0}\n".format(second))
                        f.write("model: {0}\n".format(models))
                        f.write("epochs: {0}\n".format(epoch))
                        f.write("folds: {0}\n".format(args.crossval))
                        #f.write("model_schema: {0}\n".format(args.model_schema))
                        f.write("batch_size: {0}\n".format(args.batch))
                        f.write("AUC: {0}\n".format(auc))
                        i = 0
                        while i < len(score):
                            f.write("Score: {0}\n".format(score[i]))
                            i += 1
                        f.close()
            
            #l'else non lo uso perchè uso solo il k-fold per avere tutto il dataset. Dalla konsole bisogna mettere il valore 
            #k-fold splitting (in genere metto 5)
            """
            else:
                output_dir = "/home/giacomo/tesi1/DNN_test"
                x_train, x_test, y_train, y_test = train_tes#score = model_score(y_test, pred)t_split(x,y,test_size=0.25,random_state=42)
                class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
                for models in args.model:
                    mkdir_p(output_dir + "/modello_{}".format(models))
                    output_dir1 = output_dir + "/modello_{}".format(models)
                    print(">>> Testing model ({0})...".format(models))
                    model = MT.ModelTemplates(models, x_train.shape[1])
                    model.fit(x,y,validation_data=(x_test,y_test),verbose=1,epochs = epoch, batch_size=args.batch, class_weight=class_weights)
                    model.save(output_dir1)
                    pred = model.predict(x_test)
                    score = model_score(y_test, pred)
                    auc = roc_auc_score(y_test, pred)
                    mp.rc('figure', figsize=(11,8), dpi=140)
                    fig0 = plt.gcf()
                    plot_roc_curve(y_test, pred)
                    plt.figure(figsize=(20,20))
                    plt.draw()
                    fig0.savefig(output_dir1+ "/roc_model_{}.png".format(models))
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
                        plot_confusion_matrix(cm,[0,1], title="Confusion matrix modello {}".format(models))
                        plt.draw()
                        plt.savefig(output_dir1 + "/no_norm_no_cross.png")
                        plt.clf()
                        np.set_printoptions(precision=2)
                        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        plot_confusion_matrix(cm_normalized,[0,1], title='Normalized confusion matrix')
                        plt.draw()
                        plt.savefig(args.output1 + "/norm_no_cross.png")
                    if not args.evaluate:
                        print(">>> Saving parameters...")
                        f = open(args.output + "/configssimple.txt", "w")
                        f.write("model: {0}\n".format(models))
                        f.write("epochs: {0}\n".format(epoch))
                        #f.write("model_schema: {0}\n".format(args.model_schema))
                        f.write("batch_size: {0}\n".format(args.batch))
                        f.write("AUC: {0}\n".format(auc))
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
        
