from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import os
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  # get rid oc tf_jenkins WARNING

dir_train = "../data/train.txt"
dir_test = "../data/test.txt"
np.random.seed(0)
tf.set_random_seed(1234)

def relu(x):
    return(np.maximum(x, 0))

def main(fout):
    # Load dataset.
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

    # Define the training inputs and train
    get_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x':x_train}, y=y_train, num_epochs=None, shuffle=True)
    get_train_input_fn_score = tf.estimator.inputs.numpy_input_fn(
        x={'x':x_train}, y=y_train, num_epochs=1, shuffle=True)
    #Predict
    get_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x':x_test}, y=y_test, num_epochs=1, shuffle=True)

    # Build 1 layer DNN with 1000.
    feature_columns = [tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
        hidden_units=[1000], n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.1),
        dropout=0.5,
    )
    #Train
    train_start = time.time()
    classifier.train(input_fn=get_train_input_fn, steps=200)
    train_end = time.time()
    print('training:')
    train_scores = classifier.evaluate(input_fn=get_train_input_fn_score)
    print(train_scores)
    print('testing:')
    test_start = time.time()
    scores = classifier.evaluate(input_fn=get_test_input_fn, steps=10)
    test_end = time.time()
    print(scores)
    #print('Accuracy (tf.estimator): {0:f}'.format(scores['accuracy']))
    fout.write('\t'.join([str(xx) for xx in [train_scores['accuracy'], train_scores['auc'], scores['accuracy'], scores['auc'],train_end - train_start, test_end - test_start]])+'\n')

    # #extract first layer outputs
    # z_train = np.dot(x_train, classifier.get_variable_value('dnn/hiddenlayer_0/kernel'))+classifier.get_variable_value('dnn/hiddenlayer_0/bias')
    # z_test = np.dot(x_test, classifier.get_variable_value('dnn/hiddenlayer_0/kernel'))+classifier.get_variable_value('dnn/hiddenlayer_0/bias')

    # np.savetxt('../data/nn_intermediate_train.txt', np.hstack((z_train,y_train.reshape(-1,1))))
    # np.savetxt('../data/nn_intermediate_test.txt', np.hstack((z_test,y_test.reshape(-1,1))))
    # np.savetxt('../data/nn_intermediate_relu_train.txt', np.hstack((relu(z_train),y_train.reshape(-1,1))))
    # np.savetxt('../data/nn_intermediate_relu_test.txt', np.hstack((relu(z_test),y_test.reshape(-1,1))))
    # np.savetxt('/Users/linglinhuang/Desktop/nn_w1.txt', classifier.get_variable_value('dnn/hiddenlayer_0/kernel'))
    # print('RELU Nonzero:',np.sum(relu(z_test)>0))


if __name__ == "__main__":
    try:
        niter = int(sys.argv[1])
        if niter < 1 or niter > 10:
            print('Illegal input: niter={}'.format(sys.argv[1]))
            sys.exit(1)
    except Exception as e:
        niter = 1
    fout = open('../data/nn_performance.txt','w')
    fout.write('train_acc\ttrain_auc\ttest_acc\ttest_auc\ttrain_time\ttest_time\n')
    for i in range(niter):
        main(fout)
