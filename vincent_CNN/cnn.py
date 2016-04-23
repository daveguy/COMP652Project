'''
Convolutional Neural Network implementation (Using Lasagne helper library for Theano)
Author: Vincent Petrella (modified from Lasagne MNIST tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html)
'''
import random, time, csv

import numpy as np
import theano
import theano.tensor as T

import lasagne


num_channels = 32
window_size = 150
num_events = 6
    
def load_data(read_numpy_file=False):
    
    if read_numpy_file==True:
        X_t = np.load('X_train_subj1_series1.npy')
        Y_t = np.load('Y_train_subj1_series1.npy')
        X_v = np.load('X_test_subj1_series1.npy')
        Y_v = np.load('Y_test_subj1_series1.npy')
    else:
        X_t, Y_t, X_v, Y_v = l.load_training_and_validation("features/*")
            
    return X_t, Y_t, X_v, Y_v


def build_cnn(input_var=None, dropoutRate=0.1):
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, window_size, num_channels),
                                        input_var=input_var)
  
    # First Convolution layer, convolutes over time only (5-points in time)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=4, filter_size=(1, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
   
    # Max-pooling layer of factor 2 in the time dimension: 'average_exc_pad' for mean pooling (extremely slow.. Theano bug ?)
    network = lasagne.layers.Pool2DLayer(network, pool_size=(1, 2), mode='max')
   
    # Second Conv layer, conv over freq and time ((p,3)-points)
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=2, filter_size=(1, 2),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    
    # Max-pooling layer of factor 2 in the time dimension: 'average_exc_pad' for mean pooling (extremely slow.. Theano bug ?)
    network = lasagne.layers.Pool2DLayer(network, pool_size=(1, 2), mode='max')
    
    # And, finally, fully connected 2-unit output layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=dropoutRate),
            num_units=num_events,
            nonlinearity=lasagne.nonlinearities.softmax)
        
    return network
    
    
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

        
def train_CNN(X_train,Y_train,X_val,Y_val,num_epochs):
    print "Validation Data size: " + str(Y_val.shape[0]) + " entries."

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss,prediction], updates=updates)

    #Prediction Function
    predict_fn = theano.function([input_var],[T.argmax(test_prediction, axis=1)])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        #We chose a minibatch size of 500 entries
        for batch in iterate_minibatches(X_train, Y_train, 100, shuffle=True):
            inputs, targets = batch
            t = train_fn(inputs,targets)
            train_err += t[0]
            train_batches += 1

        # And a full pass over the validation data:
        # Here we compute the number of True Positive and True negative
        # To then calculate sensitivity and specificity below
        val_acc = 0.1
        val_tpos = 0.1
        val_tneg = 0.1
        val_pred = predict_fn(X_val)[0]
        for i in range(val_pred.shape[0]):
            if val_pred[i] == Y_val[i]:
                val_acc += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation accuracy:\t\t{:.2f} %".format((val_acc / float(Y_val.shape[0])) * 100))

    # Optionally, you could now dump the network weights to a file like this:
    #np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)
    
if __name__ == '__main__':
    
    X_train, Y_train, X_val, Y_val = load_data(read_numpy_file=True)
    train_CNN(X_train, Y_train, X_val, Y_val,200)