import csv
import argparse
import mnist

import numpy as np

def pickle_results(test_csv, results):
    with open(test_csv, 'w') as fp:
        writer = csv.writer(fp)
        for r in results:
            writer.writerow(r)
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='/users/eric fowler/downloads/mnist/trainingSet/',
                        help='Directory for storing input data')
    parser.add_argument('--model', type=str,
                        default='none',
                        help='Model for processing data')
    parser.add_argument('--test_data_path', type=str,
                        default='/users/eric fowler/downloads/mnist/trainingSet/',
                        help='Directory for storing input data')
    parser.add_argument('--target', type=str,
                        default='mnist',
                        choices=['carvana', 'mnist'],
                        help='Carvana or MNIST?')
    parser.add_argument('--minimize', type=str,
                        default='simple',
                        choices=['simple', 'cross'],
                        help='Simple or X-entropy?')
    parser.add_argument('--train_step', type=str,
                        default='sgd',
                        choices=['sgd', 'adam'],
                        help='SGD or Adam optimization?')
    parser.add_argument('--env', type=str,
                        default='pc',
                        choices=['pc', 'aws'],
                        help='pc or aws?')
    parser.add_argument('--sample', type=str,
                        default='0cdf5b5d0ce1_01.jpg',
                        help='Sample image file for sizing feature tensor')
    parser.add_argument('--numclasses', type=int,
                        default=10,
                        help='Carvana=16, MNIST=10')
    parser.add_argument('--num_test_images', type=int,
                        default=200000,
                        help='Number of images to test')
    parser.add_argument('--num_train_images', type=int,
                        default=200000,
                        help='Number of images to train')
    parser.add_argument('--crop', type=bool,
                        default=False,
                        help='Crop images for speed?')
    parser.add_argument('--show', type=bool,
                        default=False,
                        help='Show some images?')
    parser.add_argument('--scale', type=float,
                        default=1.0,
                        help='Scaling factor for images')
    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='Epochs')
    parser.add_argument('--chunks', type=int,
                        default=20,
                        help='Cut samples into this many chunks')
    parser.add_argument('--test_csv', type=str,
                        default='testout.csv',
                        help='File and path for storing test output file')
    parser.add_argument('--tb_dir', type=str,
                        default='/Users/eric fowler/tensorlog/',
                        help='Directory For Tensorboard log')

    return parser.parse_known_args()


import PIL.Image
def get_pixels(filename=mnist.SAMPLE_MNIST_FILE, crop =True):
    import PIL.Image
    mm = PIL.Image.open(filename)
    if crop == True:
        x0 = mm.width / 4
        y0 = mm.height / 4
        x1 = 3 * mm.width / 4
        y1 = 3 * mm.height / 4
        mm = mm.crop((x0, y0, x1, y1))
    mma = np.array(mm)
    mma = mma.flatten('F')
    return len(mma)

def read_image(path, fname, show, scale=1.0, crop=True):
   mm = PIL.Image.open(path + fname)
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