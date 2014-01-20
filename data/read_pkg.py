import cPickle
import gzip
import os
import sys
import time

import numpy


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.     

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval 
  
  
if __name__ == '__main__':
    datasets = load_data('mnist.pkl.gz')
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    print 'train_set_x---------------------------'
    print 'len=[' + str(len(train_set_x)) + ', ' + str(len(train_set_x[0])) + ']'
    #print train_set_x
    
    print 'train_set_y---------------------------'
    print 'len=' + str(len(train_set_y))
    #print train_set_y
    
    print 'valid_set_x---------------------------'
    print 'len=[' + str(len(valid_set_x)) + ', ' + str(len(valid_set_x[0])) + ']'
    #print valid_set_x
    
    print 'valid_set_y---------------------------'
    print 'len=' + str(len(valid_set_y))
    #print valid_set_y
    
    print 'test_set_x---------------------------'
    print 'len=[' + str(len(test_set_x)) + ', ' + str(len(test_set_x[0])) + ']'
    #print test_set_x
    
    print 'test_set_y---------------------------'
    print 'len=' + str(len(test_set_y))
    #print test_set_y
    
    #train_set
    fileHandle = open('train_set_x.txt','w') 
    for i in range(0,len(train_set_x)):#50000 too long
    #for i in range(0,500):
      for j in range(0,len(train_set_x[0])):
	fileHandle.write(str(train_set_x[i][j])+'\t')
      fileHandle.write('\n')
    fileHandle.close()
    
    fileHandle = open('train_set_y.txt','w') 
    for i in range(0,len(train_set_y)):
      fileHandle.write(str(train_set_y[i])+'\n')
    fileHandle.close()
    
    #valid_set
    fileHandle = open('valid_set_x.txt','w') 
    for i in range(0,len(valid_set_x)):#50000 too long
    #for i in range(0,500):
      for j in range(0,len(valid_set_x[0])):
	fileHandle.write(str(valid_set_x[i][j])+'\t')
      fileHandle.write('\n')
    fileHandle.close()
    
    fileHandle = open('valid_set_y.txt','w') 
    for i in range(0,len(valid_set_y)):
      fileHandle.write(str(valid_set_y[i])+'\n')
    fileHandle.close()
    
    
    #test_set
    fileHandle = open('test_set_x.txt','w') 
    for i in range(0,len(test_set_x)):#10000 too long
    #for i in range(0,500):
      for j in range(0,len(test_set_x[0])):
	fileHandle.write(str(test_set_x[i][j])+'\t')
      fileHandle.write('\n')
    fileHandle.close()
    
    fileHandle = open('test_set_y.txt','w') 
    for i in range(0,len(test_set_y)):
      fileHandle.write(str(test_set_y[i])+'\n')
    fileHandle.close()
    