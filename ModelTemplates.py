from keras.models import Sequential, load_model
from keras.activations import relu
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam, SGD
from keras import optimizers
from keras import metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten, Convolution1D
from keras.initializers import random_uniform
import numpy as np

def ModelTemplates(id, input_dim):

    print(">>> Creating model...")
    model = Sequential()
    
    if id=="binary_small":
        model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
    
    if id=="binary_small2":
        model.add(Dense(units=350,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=250,activation="relu"))
        model.add(Dense(units=120,activation="relu"))
        model.add(Dense(units=70,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    if id=="binary_small3":
        model.add(Dense(units=200,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=150,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    if id=="binary_small4":
        model.add(Dense(units=200,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=170,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=150,activation="relu"))
        model.add(Dense(units=125,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
    
    if id=="binary_small5":
        model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.70))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.70))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.70))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    if id=="binary_small6":
        model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dense(units=150,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    elif id=="binary_big":
        model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    elif id=="binary_big_4":
        model.add(Dense(units=500,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=400,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        

    elif id=="binary_bigbig":
        model.add(Dense(units=800,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=500,activation="relu"))
        model.add(Dense(units=250,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
 
    elif id=="binary_bigbig_6":
        model.add(Dense(units=800,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=500,activation="relu"))
        model.add(Dense(units=500,activation="relu"))
        model.add(Dense(units=250,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    elif id=="binary_dropout":
        model.add(Dense(units=200,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])

    elif id=="binary_dropout_small":
        model.add(Dense(units=300,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.40))
        model.add(Dense(units=200,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.20))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.10))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])


    elif id=="binary_dropout2":
        model.add(Dense(units=350,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=250,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])

    elif id=="binary_dropout3":
        model.add(Dense(units=250,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=70,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=30,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
        
    elif id=="binary_dropout4":
        model.add(Dense(units=350,input_dim=input_dim, activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=250,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=50,activation="relu"))
        model.add(Dropout(0.30))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        

    elif id=="binary_best":
        model.add(Dense(units = 10 ,input_dim=input_dim, activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(units=112,activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(units=112,activation="relu"))
        #model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        
    elif id=="binary_10_10":
        model.add(Dense(units = 10 ,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=100,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
        

    return model



#grid manual search
def ModelProva(id, input_dim, done, dtwo):

    print(">>> Creating model...")
    model = Sequential()
    dtwo = int(dtwo)
    if id=="prova":
        model.add(Dense(units=50,input_dim=input_dim, activation="relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(units=done, activation="relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(units=dtwo, activation="relu"))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=50, activation = 'relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(units=1, activation='sigmoid'))
        
        opt = Adam(lr=7.3e-5)
        model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=[metrics.binary_accuracy])
    return model

def Convolutional(input_dim):
    
    print(">>> Creating model...")
    model = Sequential()
    print(input_dim)
    SEED = 42
    np.random.seed(SEED)
    #hyperparameters
    input_dimension = 900
    learning_rate = 0.0025
    momentum = 0.85
    hidden_initializer = random_uniform(seed=SEED)
    dropout_rate = 0.1
    model = Sequential()
    model.add(Convolution1D(nb_filter=100, filter_length=3, input_shape=(17,1), activation='relu'))
    model.add(Convolution1D(nb_filter=100, filter_length=2, activation='relu'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(100, input_dim=input_dimension, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    """
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout_rate))
    """
    model.add(Dense(1, activation='sigmoid'))
    """
    sgd = SGD(lr=learning_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=[metrics.binary_accuracy])
    """
    opt = Adam(lr=5e-4, decay = 4e-7)
    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=[metrics.binary_accuracy])
    
    return model

def ModelProva2(id, input_dim, done, dtwo, dthree):

    print(">>> Creating model...")
    model = Sequential()
    dtwo = int(dtwo)
    if id=="prova":
        model.add(Dense(units=done,input_dim=input_dim, activation="relu"))
        model.add(Dense(units=dtwo,activation="relu"))
        model.add(Dense(units=dthree,activation="relu"))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer="adam",
                    metrics=[metrics.binary_accuracy])
    return model
    
