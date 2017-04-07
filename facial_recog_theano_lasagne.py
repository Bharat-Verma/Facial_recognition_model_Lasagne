# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on Thu Feb 23 13:27:54 2017

@author: Bharat
"""
import os, theano, cv2, time
import numpy as np
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.nonlinearities import softmax
from lasagne.layers import ElemwiseMergeLayer

num_of_classes = -1 #total number of subjects (people)

def read_img(path, colorspace='bgr', normalize=True):
    img = cv2.imread(path)
    if colorspace == 'gray' and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif colorspace == 'bgr' and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if normalize:
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        img = img.astype(dtype=np.float32)/256.0
    return img
    
def load_data_to_GPU(data):
    
    return T._shared(np.asarray(data, dtype=theano.config.floatX), borrow=True)
    
    
def load_label_to_GPU(labels):
    
    shared_y = T._shared(np.asarray(labels, dtype=theano.config.floatX), borrow=True)

    return T.cast(shared_y, 'int32')  
    
    
def VGG_16(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 64, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 128, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 128, 3, pad=1)#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 256, 3, pad=1)#, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 256, 3, pad=1)#, flip_filters=False)
    net['conv3_3'] = ConvLayer(
        net['conv3_2'], 256, 3, pad=1)#, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_3'], 2)
    net['conv4_1'] = ConvLayer(
        net['pool3'], 512, 3, pad=1)#, flip_filters=False)
    net['conv4_2'] = ConvLayer(
        net['conv4_1'], 512, 3, pad=1)#, flip_filters=False)
    net['conv4_3'] = ConvLayer(
        net['conv4_2'], 512, 3, pad=1)#, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_3'], 2)
    net['conv5_1'] = ConvLayer(
        net['pool4'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_2'] = ConvLayer(
        net['conv5_1'], 512, 3, pad=1)#, flip_filters=False)
    net['conv5_3'] = ConvLayer(
        net['conv5_2'], 512, 3, pad=1)#, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_3'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=4096)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net



def Tiny_VGG(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 32, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 32, 3, pad=1)#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(
        net['pool2'], 32, 3, pad=1)#, flip_filters=False)
    net['conv3_2'] = ConvLayer(
        net['conv3_1'], 32, 3, pad=1)#, flip_filters=False)
#    net['conv3_3'] = ConvLayer(
#        net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_2'], 2)
    net['fc6'] = DenseLayer(net['pool3'], num_units=256)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)
#    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=1024)
#    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0)
    net['fc8'] = DenseLayer(
        net['fc6_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
    
def Deep_ID2(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 20, 3, pad=0)#, flip_filters=False)
#    net['conv1_2'] = ConvLayer(
#        net['conv1_1'], 64, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_1'], 2)
    
    net['conv2_1'] = ConvLayer(
        net['pool1'], 40, 4, pad=0)#, flip_filters=False)
#    net['conv2_2'] = ConvLayer(
#        net['conv2_1'], 32, 3, pad=1)#, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_1'], 2)
    
    net['conv3_1'] = ConvLayer(
        net['pool2'], 60, 3, pad=0)#, flip_filters=False)
#    net['conv3_2'] = ConvLayer(
#        net['conv3_1'], 16, 3, pad=1)#, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_1'], 2)
    
    net['conv4_1'] = ConvLayer(
        net['pool3'], 80, 2, pad=0)#, flip_filters=False)
#    net['conv4_2'] = ConvLayer(
#        net['conv4_1'], 8, 3, pad=1)#, flip_filters=False)

    net['fc6'] = DenseLayer(net['conv4_1'], num_units=160)
#    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)

    net['fc7'] = DenseLayer(net['pool3'], num_units=160)
#    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0)
    
      
    net['merge'] = ElemwiseMergeLayer([net['fc6'],net['fc7']],merge_function=theano.tensor.add)
  
    net['fc8'] = DenseLayer(
        net['merge'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    

    return net



def TinyTiny_VGG(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['pool1'], 32, 3, pad=1)#, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 32, 3, pad=1)#, flip_filters=False)
    net['    '] = PoolLayer(net['conv2_2'], 2)
    net['fc6'] = DenseLayer(net['pool2'], num_units=512)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=512)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net 
    
def Fully_Conv(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 4, (2,2), pad=0, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 64, 4, pad=0, flip_filters=False)
#    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(
        net['conv1_2'], 128, 6, (2,2), pad=0, flip_filters=False)
    net['conv2_2'] = ConvLayer(
        net['conv2_1'], 256, 6, (2,2), pad=0, flip_filters=False)
#    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['fc6'] = DenseLayer(net['conv2_2'], num_units=1024)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=512)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net    
    

def Two_layer(input_var=None):
    net = {}
    net['input'] = InputLayer(shape=(None, 3, 64, 64),input_var=input_var)
    net['conv1_1'] = ConvLayer(
        net['input'], 32, 3, pad=1)#, flip_filters=False)
    net['conv1_2'] = ConvLayer(
        net['conv1_1'], 32, 3, pad=1)#, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['fc6'] = DenseLayer(net['pool1'], num_units=512)
    net['fc6_dropout'] = DropoutLayer(net['fc6'], p=0)
    net['fc7'] = DenseLayer(net['fc6_dropout'], num_units=512)
    net['fc7_dropout'] = DropoutLayer(net['fc7'], p=0)
    net['fc8'] = DenseLayer(
        net['fc7_dropout'], num_units=num_of_classes, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net

    
if __name__ == '__main__':
    directory = '/home/bverma/Documents/back-up_cropped_pics/Data_set_64x64'
    data = {0:{}}
    image_per_direction = 10
    labels = []
    seed = 22345 #seed for the random state
    #read the images from the directory
    for subdir, dirs, files in os.walk(directory):
        for image in files:
            r = read_img(directory+'/'+image)
            s = image.split('_')
            if int(s[1]) not in labels:
                labels.append(int(s[1]))
                
                
            #if s[0][-2:]=='00':
            try:
                data[0][int(s[1])].append([r,int(s[1])])
            except KeyError:
                data[0][int(s[1])] = [[r,int(s[1])]]
   
    
    
    num_of_classes = len(labels)
    print num_of_classes, labels
    ########################################################
    np.random.RandomState(seed)
    train_set = list(); test_set = list()
    #give the real label to the list
    for k,v in data.iteritems():
        for k2, v2 in v.iteritems():
            temp = []
            for item in data[k][k2]:
                temp.append((item[0],labels.index(item[1])))
            np.random.shuffle(temp)    
            data[k][k2] = temp
            for image_tuple in data[k][k2][:image_per_direction]:
                test_set.append(image_tuple)
            for image_tuple in data[k][k2][image_per_direction:]:
                train_set.append(image_tuple)    
                
    np.random.RandomState(seed)
    np.random.shuffle(train_set)
    np.random.shuffle(test_set) 
    print (len(test_set))
    print (len(train_set))  
    #########################################     
    #Some hyper-parameter for the training/testing sets
    mini_batch_size = 10; num_train_batches = len(train_set)/mini_batch_size 
    num_test_batches = len(test_set)/mini_batch_size
    
    #training_data and training_labels are two numpy arrays
    training_data = np.asarray([image_tuple[0].transpose(2,0,1).reshape(3,64,64) for image_tuple in train_set])
    training_labels = np.asarray([image_tuple[1] for image_tuple in train_set]) 

    testing_data = np.asarray([image_tuple[0].transpose(2,0,1).reshape(3,64,64) for image_tuple in test_set])
    testing_labels = np.asarray([image_tuple[1] for image_tuple in test_set])
    
    training_set_x = load_data_to_GPU(training_data)
    training_set_y = load_label_to_GPU(training_labels)
    
    testing_set_x = load_data_to_GPU(testing_data)
    testing_set_y = load_label_to_GPU(testing_labels)
    
    print 'Upload testing data to CPU...'
#    test_set_x,test_set_y = load_data_to_GPU(testing_data[:mini_batch_size],testing_labels[:mini_batch_size])
    
#    train_set_x,train_set_y = load_data_to_GPU(training_data[:mini_batch_size], training_labels[:mini_batch_size])

    # Prepare Theano variables for inputs and targets
    X = T.tensor4('x')
    y = T.ivector('y')
    index = T.lscalar()

    X = X.reshape((mini_batch_size,3,64,64))

#    network = VGG_16(X)
#    network = Tiny_VGG(X)
#    network = TinyTiny_VGG(X)
#    network = Fully_Conv(X)
#    network = Two_layer(X)
    network = Deep_ID2(X)

    prediction = lasagne.layers.get_output(network['prob'])
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network.values(), trainable=True)

    
#    updates = lasagne.updates.sgd(
#            loss, params, learning_rate=0.01)
            
    updates = lasagne.updates.adagrad(
           loss, params, learning_rate=0.01)
            
#    updates = lasagne.updates.adam(loss, params)        
            
    test_prediction = lasagne.layers.get_output(network['prob'], deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            y)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), y),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([index], loss, updates=updates, givens={X:training_set_x[index*mini_batch_size:(index+1) * mini_batch_size], 
                                                                       y:training_set_y[index*mini_batch_size:(index+1)*mini_batch_size]}, on_unused_input='ignore')
    test_fn = theano.function([index], [test_loss, test_acc], givens={X:testing_set_x[index*mini_batch_size:(index+1)*mini_batch_size],
                                                                      y:testing_set_y[index*mini_batch_size:(index+1)*mini_batch_size]}, on_unused_input='ignore')
    
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    best_test_acc = -1
    for epoch in range(40):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for i in range(num_train_batches):
            if (i+1) % 20 == 0:
                print 'training at {} epoch and {} iteration'.format(epoch+1,i+1)
                                                                  
            train_err += train_fn(i)
            train_batches += 1
            
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        test_err = 0
        test_acc = 0
        test_batches = 0                                           
        
        for i in range(num_test_batches):
            err,acc = test_fn(i)
            test_err += err
            test_acc += acc
            test_batches += 1
        
        current_test_acc = test_acc/test_batches * 100
        print "  test loss:\t\t\t{:.6f}".format(test_err/test_batches)
        print "  test accuracy:\t\t{:.2f} %".format(current_test_acc)
        if current_test_acc > best_test_acc:
            print 'Saving the network...'
            np.savez('Fully.npz', *lasagne.layers.get_all_param_values(network.values()))
            print 'Network saved!'
            best_test_acc = current_test_acc
    print 'Done! The best testing accuracy is '+str(best_test_acc)+'%.'                                               
