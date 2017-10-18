import numpy as np
import tensorflow as tf
import os
from PIL import  Image
import random
import matplotlib.pyplot as plt

file_path_test = '/Users/Eric Fowler/Downloads/carvana/test/'
file_path_train = '/Users/Eric Fowler/Downloads/carvana/train/'
file_path_mnist_test= '/Users/Eric Fowler/Downloads/mnist/testSet/'
file_path_mnist_train= '/Users/Eric Fowler/Downloads/mnist/trainingSet/'
SAMPLE_FILE_MNIST = '/Users/Eric Fowler/Downloads/mnist/trainingSet/0/img_1.jpg'
SAMPLE_FILE = file_path_train+'0cdf5b5d0ce1_01.jpg'

def get_pixels(filename=SAMPLE_FILE,crop =True):
    mm = Image.open(filename)
    if crop == True:
        x0 = mm.width / 4
        y0 = mm.height / 4
        x1 = 3 * mm.width / 4
        y1 = 3 * mm.height / 4
        mm = mm.crop((x0, y0, x1, y1))
    mma = np.array(mm)
    mma = mma.flatten('F')
    return len(mma)


def get_mnist_tensor_list(numclasses = 10,path=file_path_mnist_train, num=None):

    label_pool=[]
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


def get_tensor_list(numclasses=16,path='/Users/Eric Fowler/Downloads/carvana/train/', num=None):

    files = os.listdir(path)

    if files == []:
        return None

    jpgs = [f for f in files if f.endswith('jpg')]
    nums = [m.split('_')[1] for m in jpgs]
    nums = [n.split('.')[0] for n in nums]
    nums = np.asarray(nums, dtype=np.int32) - 1
    labels = np.zeros((len(nums), numclasses))
    labels[np.arange(len(nums)), nums] = 1

    if num == None:
        num = len(jpgs)

    return (list(zip(jpgs[:num], labels[:num])))

def read_image(path, fname, show, scale=1.0, crop=True):
   mm = Image.open(path + fname)
   if crop==True:
       x0=mm.width/4
       y0=mm.height/4
       x1 =3*mm.width/4
       y1 = 3*mm.height/4
       mm=mm.crop((x0,y0,x1,y1))

   if scale != 1.0:
       mm = mm.resize((int(mm.size[0]/scale), int(mm.size[1]/scale)))

   if show == True:
       plt.imshow(mm)
       plt.show()

   #mm = mm.convert('F')
   mma = np.array(mm)
   mma = mma.flatten('F')
   return mma
#train(tr_list=zz[0],train_step=train_step,epochs=EPOCHS,numclasses=NUMCLASSES,sess=sess,x=x,y_=y_,crop=CROP,show=SHOW,scale=SCALE)
def train(train_step, sess, tr_list,x,y_,epochs,numclasses,show,crop,scale=1.0):
    idx=0
    for iter in range(epochs):
        train_features = []
        train_labels = []
        for filename, labels_onehot in tr_list:
            showChild = False and show
            if idx >= 0 and idx < numclasses:
                showChild = True and show
            if idx % 100 == 0:
                print('Training:', idx,iter)
                if idx >= 0 and idx < numclasses:
                    showChild = True and show

            train_features.append(read_image(file_path_mnist_train, filename, scale=scale, show=showChild,crop=crop))
            train_labels.append(labels_onehot)
            #take # of pixels from size of 1st image
            if idx == 0:
                NUMPIXELS=len(train_features[0])

            idx+=1
        train_step.run(feed_dict={x: train_features, y_: train_labels})


def test(tt_list, sess, accuracy,x,y_,scale,crop,show):
    idx = 0

    test_features = []
    test_labels = []
    for filename, labels_onehot in tt_list:
        if idx % 100 == 0:
            print('Testing:', idx)

        test_features.append(read_image(file_path_mnist_train, filename, scale=scale, crop=crop,show=show))
        test_labels.append(labels_onehot)
        idx += 1

    test_result = sess.run(accuracy, feed_dict={x: test_features, y_: test_labels})
    print('Testing:%d (%02.12f)' % (idx - 1, test_result))
# resize  same
# Create the model
# set up to feed an array of images [images, size_of_image]
def make_graph(numpixels, numclasses):
    x = tf.placeholder(tf.float32, [None,numpixels])

    #variables for computation
    #2d array of weights,[pixels, classes]
    W = tf.Variable(tf.zeros([numpixels,numclasses]))
    #W = tf.Variable(tf.zeros([NUMPIXELS,NUMCLASSES],dtype=tf.float32),dtype=tf.float32)

    #1d array of bias vars
    b = tf.Variable(tf.zeros(numclasses))
    #b = tf.Variable(tf.zeros(NUMCLASSES,dtype=tf.float32),dtype=tf.float32)
    #the array of 'answers' produced by fxn. of W, x & b, a 1xNUMCLASSES array
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Define loss and optimizer..why is this 2d?
    y_ = tf.placeholder(tf.float32, [None,numclasses])

    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    #train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run(session=sess)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x,y,y_,train_step,sess,accuracy

def main():
    NUMDIGITS = 10
    NUMCLASSES = NUMDIGITS
    CROP = False
    SHOW=False
#    NUMCLASSES = 16
    NUMPIXELS = -1
    SCALE = 1.0
    EPOCHS = 1
    tr=get_mnist_tensor_list(numclasses=NUMCLASSES,path=file_path_mnist_train,num=200000)
    random.shuffle(tr)
    tr_len = int(len(tr))
    tr_list = tr[:int(7*tr_len/8)]
    tt_list = tr[int(7*tr_len/8):]
    tr=None
    NUMPIXELS=get_pixels(crop=CROP,filename=SAMPLE_FILE_MNIST)

    x,y,y_,train_step,sess,accuracy=make_graph(NUMPIXELS,NUMCLASSES)

    trainer = [tr_list[i:i+len(tr_list)//20] for i in range(0,len(tr_list),len(tr_list)//20)]
    tester=[tt_list[i:i+len(tt_list)//20] for i in range(0,len(tt_list),len(tt_list)//20)]
    z= zip(trainer,tester)
    for zz in z:
        train(tr_list=zz[0],train_step=train_step,epochs=EPOCHS,numclasses=NUMCLASSES,sess=sess,x=x,y_=y_,crop=CROP,show=SHOW,scale=SCALE)
        test(tt_list=zz[1],sess=sess,accuracy=accuracy,x=x,y_=y_,scale=SCALE, crop=CROP,show=SHOW)



if __name__ == '__main__':
    main()




