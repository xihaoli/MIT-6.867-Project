import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf
import itertools
from sklearn.metrics import roc_auc_score
# read data
dir_train = "../data/train.txt"
dir_test = "../data/test.txt"
np.random.seed(0)
tf.set_random_seed(1234)
with open(dir_train) as f:
  colnames = f.readline().split('\t')
  ncols = len(colnames)
data = np.loadtxt(dir_train, delimiter='\t', skiprows=1, usecols=range(1,ncols+1))
data_test = np.loadtxt(dir_test, delimiter='\t', skiprows=1, usecols=range(1,ncols+1))
x_train = data[:, :(data.shape[1]-1)]
x_test = data_test[:, :(data_test.shape[1]-1)]
y_train = data[:, data.shape[1]-1].astype(int)
y_test = data_test[:, data_test.shape[1]-1].astype(int)
not_const = np.std(x_train, 0)!=0
x_train = x_train[:, not_const]
x_test = x_test[:, not_const]


es = [True]#early stopping
nunits_1 = [100, 1000, 2000]#number of units in hidden layer 1
nunits_23 = [0, 100]#number of units in hidden layer 2 or 3
dropouts = [0.5] #dropout rates
bn = [True]#batch normalization
lr = [0.01, 0.1] #learning rates
# activation = ['sigmoid', 'relu']#activation functions
activation = ['relu']#activation functions
# optimizer = ['SGD', 'Adagrad']#optimizer in compiling
optimizer = ['Adagrad']#optimizer in compiling

combinations = list(itertools.product(es, nunits_1, nunits_23, nunits_23, dropouts, bn, lr, activation, optimizer)).append((False, 1000, 0, 0, 0.5, False, 0.1, 'relu', 'Adagrad'))
len(combinations)

fout = open('../data/nn_results.csv', 'w')
fout.write('EarlyStopping,Units1,Units2,Units3,Dropout,BatchNormalization,LearningRate,Activation,Optimizer,TestLoss,TestAcc,TestAUC,TrainAcc,ValAcc\n')
fout.close()
for i in range(5): #run 5 iterations
    fout = open('../data/nn_results.csv', 'a')
    for j,combination in enumerate(combinations):
        print('{}th, {}/{}, comb={}'.format(i+1, j+1, len(combinations), combination))
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model = Sequential()
        # hidden layer 1
        model.add(Dense(units=combination[1], activation=combination[-2], input_dim=x_train.shape[1], init='RandomNormal'))
        model.add(Dropout(rate=0.5))
        model.add(BatchNormalization())
        if combination[2] > 0:
            # hidden layer 2
            model.add(Dense(units=combination[2], activation=combination[-2], input_dim=x_train.shape[1]))
            model.add(Dropout(rate=0.5))
            model.add(BatchNormalization())
            if combination[3] > 0:
                # hidden layer 3
                model.add(Dense(units=combination[3], activation=combination[-2], input_dim=x_train.shape[1]))
                model.add(Dropout(rate=0.5))
                model.add(BatchNormalization())
        model.add(Dense(units=1, activation='sigmoid'))
        if combination[-1] == 'SGD':
            model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=combination[-3], momentum=0.9),
                  metrics=['accuracy'])
        elif combination[-1] == 'Adagrad':
            model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adagrad(lr=combination[-3], epsilon=1e-08),
                  metrics=['accuracy'])
        hist = model.fit(x_train, y_train, epochs=50, shuffle=True, validation_split=0.33, callbacks=[early_stopping], verbose=0)
        loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
        y_pred = model.predict_proba(x_test, verbose=0)
        auc = roc_auc_score(y_test, y_pred)
        # print(loss_and_metrics)
        fout.write(','.join([str(x) for x in combination + tuple(loss_and_metrics) + (auc, hist.history['acc'][-1], hist.history['val_acc'][-1])])+'\n')
    fout.close()
