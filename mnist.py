import numpy as np
import os

file_path_mnist_test= '/Users/Eric Fowler/Downloads/mnist/testSet/'
file_path_mnist_train= '/Users/Eric Fowler/Downloads/mnist/trainingSet/'
SAMPLE_MNIST_FILE = '/Users/Eric Fowler/Downloads/mnist/trainingSet/0/img_1.jpg'

def get_mnist_train_tensor_list(path, numclasses = 10, num=None):
    files=[]
    labels=[]
    for x in range(numclasses):
        label = np.zeros(numclasses)
        label[x]= 1
        fpath = path+str(x)+'/'
        jpgs = [f for f in os.listdir(fpath) if f.endswith('jpg') or f.endswith('jpeg')]
        for j in jpgs:
            files.append(str(x)+'/'+j)
            labels.append(label)

    if num == None:
        num = len(files)

    return (list(zip(files[:num], labels[:num])))


def mnist_clean(l,n):
    for file,label in l:
        file=file.rstrip(['0123456789//'])
        file = file + '//'