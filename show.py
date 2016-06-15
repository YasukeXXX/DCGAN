import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import Variable
from chainer import serializers
import os

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir','-o',default = None)
parser.add_argument('--input_dir','-i',default = None)
parser.add_argument('--input_epoch','-e',type=int,default = None)
args = parser.parse_args()

# from mnist_fc import Generator,Discriminator
# from mnist_conv import Generator,Discriminator
from people_conv import Generator,Discriminator

G = Generator()
len_z = G.in_size

if args.input_dir != None:
	in_dir = args.input_dir
	if args.input_epoch != None:
		serializers.load_hdf5("%s/gan_model_gen%d.h5"%(in_dir, args.input_epoch), G)
	else:
		serializers.load_hdf5("%s/current_gen.h5"%(in_dir), G)
batchsize = 25
z = Variable(np.random.uniform(-1,1,(batchsize,len_z)).astype(np.float32))
y1 = G(z,False)
fig = plt.figure()
ax = []
for i in xrange(batchsize):
	ax.append(fig.add_subplot(5,5,i+1))
	ax[i].imshow(np.array(y1.data[i]).reshape(G.imshape[1],G.imshape[2]),cmap='gray')
	ax[i].axis('off')

class callback(object):
	def suffle(self,event):
		z = Variable(np.random.uniform(-1,1,(batchsize,len_z)).astype(np.float32))
		y1 = G(z,False)
		for i in xrange(batchsize):
			ax[i].imshow(np.array(y1.data[i]).reshape(G.imshape[1],G.imshape[2]),cmap='gray')
		plt.draw()
c = callback()

axsuffle = plt.axes([0.8, 0.01, 0.1, 0.075])
button = Button(axsuffle, 'Suffle')
button.on_clicked(c.suffle)
plt.show()
