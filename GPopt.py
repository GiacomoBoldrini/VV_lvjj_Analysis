import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import math
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args
from sklearn import metrics
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import KFold
from scipy import interp
import ModelTemplates as MT
from keras.callbacks import *
from copy import deepcopy
from keras import metrics
import DNN_Plotter
import random
random.seed(10)
import ROOT as r

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--crossval', type=int, required=False, help="K-Fold cross validation number") 
#parser.add_argument('-e', '--epochs', type=int, required=False, help = "number of epochs for fit model")
#parser.add_argument('-bs', '--batch', type=int, required=True, help = "batch size (1024 or 32)")
#parser.add_argument('-t', '--type', required=True, help = "dimensions type", action = "store_true")
args = parser.parse_args()


###################### USEFUL FUNCTIONS ############################
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
    #calulating threshold predisctions
    i=0
    while i<len(pred):
        if pred[i]>th:
            pred[i]=1;
        else:
            pred[i]=0;
        i=i+1
    #defining the score just on 1 jets since we wanrt all of them
    pred = np.concatenate(pred)
    print(pred)
    print(y_test)
    print(len(pred))
    print(len(y_test))
    j = 0
    h = 0
    count = 0
    while j < len(pred):
        if y_test[j] == 1:
            count+=1
            if pred[j] == 1:
                h += 1
        j += 1
    #return accuracy of real jets 1 
    return h/count

def plot_conf_matrix(oos_test, pred, th, output_dir1):
    print(">>>>Confusion matrix th = {}".format(th))
    i = 0 
    while i < len(pred):
        if pred[i]>th:
            pred[i]=1
        else:
            pred[i]=0
        i=i+1
                            
    DNN_Plotter.Confusion_matrix_best(output_dir1 + "/ConfusionMatrix",oos_test, pred, th)
    plt.clf()
    plt.close()
###################### DIMENSION OF SEARCHING SPACE ############################


dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',name='learning_rate')

dim_num_dense_layers = Integer(low=1, high=20, name='num_dense_layers')

dim_num_dense_nodes = Integer(low=1, high=700, name='num_dense_nodes')

dim_activation = Categorical(categories=['relu', 'sigmoid', "tanh", "linear"],
                             name='activation')

dim_dropout = Real(low= 0.0, high= 0.99, name='dropout_rate')

#dim_decay = Real(low= 1e-9, high= 10, name='decay_rate')

dim_batch = Integer(low= 50, high= 10000, name='batch_dimension')

#dim_first_layer = Integer( low= 1, high=50, name='num_first_layer')

"""
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_dropout,
              dim_decay,
              dim_batch,
              dim_first_layer]
"""
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch,
              dim_dropout]
###################### INPUT DATA, ACC, PATHS ############################

data = np.load("/home/giacomo/tesi1/VBSAnalysis/Data.npy")
print(data.shape)
count0 =0
count1 = 0
for i in data[:,17]:
    if i == 0:
        count0 += 1
    else:
        count1 += 1
print(">>>Getti 0 prima shuffle: {}".format(count0))
print(">>>Getti 1 dopo shuffle: {}".format(count1))
np.random.shuffle(data)

#data = data[:100,:]

print(data.shape)
input_shape = data.shape[1] -1 
x,y = to_xy(data, 17)
count0 =0
count1 = 0
for i in y:
    if i == 0.:
        count0 +=1
    else:
        count1 +=1
print(">>>Rateo 0 su 1: {}".format(count0/count1))
print(">>>Getti 0 dopo shuffle: {}".format(count0))
print(">>>GEtti 1 dopo shuffle: {}".format(count1))
path_best_model = "/home/giacomo/tesi1/DNN_test/best_model/best_gp_network"
best_accuracy = 0.0
global mycount
mycount = 0

###################### MODEL CREATION ############################

def Modeloptimizer(learning_rate,num_dense_layers, num_dense_nodes, activation, batch_dimension, dropout_rate):
    print(">>> Creating model...")
    model = Sequential()
    print(">>> input dim {}".format(input_shape))
    model.add(Dense(units = 35 ,input_dim=input_shape, activation="relu"))
    model.add(Dropout(dropout_rate))
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)

        # Add the dense / fully-connected layer to the model.
        # This has two hyper-parameters we want to optimize:
        # The number of nodes and the activation function.
        model.add(Dense(units=num_dense_nodes,
                        activation=activation,
                        name=name))
        model.add(Dropout(dropout_rate))
        
    #last layer for categorical crossentropy must be one, activation sigmoid
    model.add(Dense(units=1, activation='sigmoid'))
    #optimizer = Adam(lr=learning_rate, decay = decay_rate)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    return model


"""
def Modeloptimizer(num_dense_layers, num_dense_nodes, activation):
    print(">>> Creating model...")
    model = Sequential()
    print(">>> input dim {}".format(input_shape))
    model.add(Dense(units = 10 ,input_dim=input_shape, activation="relu"))
    model.add(Dense(units=112,activation="sigmoid"))
    model.add(Dense(units=112,activation="sigmoid"))
    model.add(Dense(units=1, activation='sigmoid'))
    #last layer for categorical crossentropy must be one, activation sigmoid
    model.add(Dense(units=1, activation='sigmoid'))
    decay_rate = learning_rate/args.epochs
    optimizer = Adam(lr = learning_rate, decay = decay_rate)
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=[metrics.binary_accuracy])


    return model
"""

###################### FITNESS FUNCTION FOR BEST ACC ############################

@use_named_args(dimensions=dimensions)
def fitness( learning_rate, num_dense_layers, num_dense_nodes, activation, batch_dimension, dropout_rate):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    global mycount
    print("counter cycle: {}".format(mycount))
    
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print('dropout_rate:', dropout_rate)
    #print('decay rate:', decay_rate)
    print('batch dimension:', batch_dimension)
    #print('neurons first layer: ', num_first_layer)
    print()
    
    
    # Create the neural network with these hyper-parameters.
    class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
    kf = KFold(3)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=15, verbose=1, mode='auto', baseline=None)
    fold = 0
    oos_y = []
    oos_pred = []
    print(kf)
    for train,test in kf.split(x):
        fold+=1
        print(" Fold #{}".format(fold))
        #csv_logger = CSVLogger(output_dir1 +'/training_{0}.log'.format(fold))
        x_train = x[train]
        y_train = y[train]
        x_test = x[test]
        y_test = y[test]
        model = Modeloptimizer(learning_rate=learning_rate, num_dense_layers=num_dense_layers,num_dense_nodes=num_dense_nodes,activation=activation, batch_dimension= batch_dimension, dropout_rate = dropout_rate)
        #model=Modeloptimizer(num_dense_layers=num_dense_layers,num_dense_nodes=num_dense_nodes,activation=activation, batch_dimension= batch_dimension, num_first_layer = num_first_layer)
        #history = model.fit(x_train,y_train,v
        model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = 3000, batch_size=batch_dimension, class_weight=class_weights,shuffle=True, callbacks=[early_stop])
        #model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=1, epochs = 15, batch_size=1000, class_weight=class_weights,shuffle=True, callbacks=[early_stop])
        predi = model.predict(x_test)
        oos_y.append(y_test)
        oos_pred.append(predi)
        
    oos_y = np.concatenate(oos_y)
    os_pred = np.concatenate(oos_pred)
    oos_pred = np.concatenate(oos_pred)
    #definisco histogramma per distribuzione della deviazione
    mycount += 1
    histo = r.TH1F("h1", "Deviance 0", 50, -1,1)
    histo.SetFillColor(r.kBlack)
    histo.SetFillStyle(3003)
    histo.SetLineColor(r.kBlack)
    histo.SetLineWidth(2)
    histo1 = r.TH1F("h1", "Deviance 1", 50, -1,1)
    histo1.SetFillColor(r.kBlue)
    histo1.SetFillStyle(3003)
    histo1.SetLineColor(r.kBlue)
    histo1.SetLineWidth(2)
    i = 0
    while i<len(oos_y):
        if oos_y[i]==0:
            histo.Fill(abs(oos_y[i]-oos_pred[i]))
        else:
            histo1.Fill(abs(oos_y[i]-oos_pred[i]))
        i +=1
    c1 = r.TCanvas("c1", "c1", 50, 50, 1000, 800)
    histo.Draw("histo")
    histo1.Draw("histo same")
    
    c1.Draw()
    
    #computo lo score
    
    th = 0.5
    score = []
    while th < 1:
        toscore = deepcopy(oos_pred)
        z = model_score(oos_y, toscore, th)
        score.append([z, th])
        th = th + 0.1
        th = round(th, 1)
    
    #We define max score for some threshold. It's not important which threshold it is
    score = sorted(score, reverse = True)


    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = score[0][0]

    # Print the classification accuracy.
    print(score)
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print("Threshold: {0}".format(score[0][1]))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        #model.save(path_best_model+ "/best_model_{0:.4f}".format(accuracy))
        #model.save(path_best_model + "/best")
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy


###################### MAIN, CALLING FUNCTIONS ############################
#print("counter cycle: {}".format(i))
#default_parameters = [2, 10, 'relu', 1024, 10]
default_parameters = [7.3e-04, 10, 10, 'relu', 1000, 0.]
search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=300,
                            x0=default_parameters,
                            )

print(">>> Final Results: ")
print()
print("BEST RESULTS: {0}".format(search_result.x))
print()
print("BEST ACCURACY: {0}".format(-search_result.fun))



plot_histogram(result=search_result,
                         dimension_name='activation')
plt.show()

                         
plot_objective_2D(result=search_result,
                        dimension_name1='num_dense_nodes',
                        dimension_name2='num_dense_layers',
                        levels=11)
plt.show()



#dim_names = ['learning_rate', 'num_dense_nodes', 'num_dense_layers', 'dropout_rate', 'decay_rate', 'batch_dimension']
dim_names = ['learning_rate','num_dense_nodes', 'num_dense_layers', 'batch_dimension', 'dropout_rate']

fig2 = plt.figure()
plot_convergence(search_result)
plt.show()
fig2.savefig(path_best_model + "/convergence.png")
plt.clf()
plt.close()

#fig, ax = plot_objective(result=search_result, dimension_names=dim_names)

fig3 = plot_objective(result=search_result, dimension_names=dim_names)
plt.show()


#fig1, ax = plot_evaluations(result=search_result, dimension_names=dim_names)
fig4 = plot_evaluations(result=search_result, dimension_names=dim_names)
plt.show()

print(">>>Saving Model: ")
best_learning_rate = search_result.x[0]
best_num_layers = search_result.x[1]
best_num_neurons = search_result.x[2]
best_activation = search_result.x[3]
#best_decay_rate = search_result.x[5]
best_batch_size = search_result.x[4]
best_dropout = search_result.x[5]
#best_first_layer = search_result.x[4]
print("best learning rate: {}".format(best_learning_rate))
print("best number of layers: {}".format(best_num_layers))
print("best number of neurons {}".format(best_num_neurons))
print("best activation function: {}".format(best_activation))
print("best dropout rate: {}".format(best_dropout))
#print(best_decay_rate)
print("best batch size: {}".format(best_batch_size))
#print(best_first_layer)
#Save_best(best_learning_rate, best_num_layers, best_num_neurons, best_activation, best_dropout,best_decay_rate, best_batch_size, best_first_layer)
#model = Save_best(best_learning_rate, best_num_layers, best_num_neurons, best_activation, best_batch_size, best_dropout)
#model.save("/home/giacomo/tesi1/DNN_test/best_model/best_data/best")  
K.clear_session()


