import tensorflow as tf
import os
import glob
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, LeakyReLU, Activation, UpSampling2D, Conv2D, MaxPooling2D
plt.switch_backend('agg')

class DCGAN(object):
	def __init__(self, width=160, height=120, channels=3):
		self.width = width
		self.height = height
		self.channels = channels

		self.shape = (self.width, self.height, self.channels)
		self.optimizer = Adam(lr=0.0002, decay=8e-9)
		self.noise_gen = np.random.normal(0, 1, (100,))
		self.G = self.generator()
		self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

		self.D = self.discriminator()
		self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
		self.stacked_G_D = self.stacked_G_D()

		self.stacked_G_D.compile(loss='binary_crossentropy', optimizer=self.optimizer)
	
	def generator(self):
		model = Sequential()
		model.add(Dense(1024, input_shape=(100,)))
		model.add(Activation('tanh'))
		model.add(Dense(128 * int(self.width/4) * int(self.height/4)))
		model.add(BatchNormalization())
		model.add(Activation('tanh'))
		model.add(Reshape((int(self.width/4), int(self.height/4), 128), input_shape=(128*int(self.width/4)*int(self.height/4),)))
		model.add(UpSampling2D(size=(2, 2)))
		model.add(Conv2D(64, (5, 5), padding='same'))
		model.add(Activation('tanh'))
		model.add(UpSampling2D(size=(2, 2)))
		model.add(Conv2D(3, (5, 5), padding='same'))
		model.add(Activation('tanh'))
		model.summary()

		return model
	
	def discriminator(self):

		model = Sequential()
		model.add(
		            Conv2D(64, (5, 5),
			    padding='same',
			    input_shape=(self.width, self.height, self.channels))
			 )
		model.add(Activation('tanh'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(128, (5, 5)))
		model.add(Activation('tanh'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation('tanh'))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		model.summary()

		return model
	
	def stacked_G_D(self):
		self.D.trainable = False

		model = Sequential()
		model.add(self.G)
		model.add(self.D)

		return model

	def train(self, x_train, epochs=20000, batch=32, save_interval=200):
		for cnt in range(epochs):
			random_index = np.random.randint(0, len(x_train) - batch/2)
			legit_images = x_train[random_index : random_index + int(batch/2)]
			legit_images = legit_images.reshape(int(batch/2), self.width, self.height, self.channels)

			gen_noise = np.random.normal(0, 1, (int(batch/2), 100))
			syntetic_images = self.G.predict(gen_noise)

			x_combined_batch = np.concatenate((legit_images, syntetic_images))
			y_combined_batch = np.concatenate((np.ones((int(batch/2), 1)), np.zeros((int(batch/2), 1))))

			d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)

			noise = np.random.normal(0, 1, (batch, 100))
			y_mislabeled = np.ones((batch, 1))
			g_loss = self.stacked_G_D.train_on_batch(noise, y_mislabeled)
			print(d_loss, g_loss)
			print('epoch: %d, [Discriminator :: d_loss: %f], [Generator :: loss %f]' % (cnt, d_loss[0], g_loss))
			if cnt % save_interval == 0:
				self.plot_images(save2file=True, step=cnt)
	
	def plot_images(self, save2file=False, samples=16, step=0):
		filename = './samples/donkey_%d.png' % step
		noise = np.random.normal(0, 1, (samples, 100))

		images = self.G.predict(noise)

		plt.figure(figsize=(10, 10))

		for i in range(images.shape[0]):
			plt.subplot(4, 4, i+1)
			image = images[i, :, :, :]
			image = np.reshape(image, [self.height, self.width])
			plt.imshow(image)
			plt.axis('off')
		plt.tight_layout()

		if save2file:
			plt.savefig(filename)
			plt.close('all')
		else:
			plt.show()


flags = tf.app.flags
flags.DEFINE_string("dataset", "donkey", "The name of dataset [donkey, mnist]")
flags.DEFINE_integer("input_height", 120, "The size of image to use. [120]")
flags.DEFINE_integer("input_width", 160, "The size of image to use. If None, same value as input_height [160]")
flags.DEFINE_integer("channels", 3, "Image channels [3]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags = flags.FLAGS

if not os.path.isdir('models'):
	os.makedirs('models')
if not os.path.isdir('samples'):
	os.makedirs('samples')

dataset = None

if flags.dataset == 'mnist':
	flags.input_height = 28
	flags.input_width = 28
	flags.channels = 1
	(dataset, _), (_, _) = tf.keras.datasets.minist.load_data()
	dataset = (dataset.astype(np.float32) - 127.5) / 127.5
elif flags.dataset == 'donkey':
	paths = glob.glob(os.path.join(flags.data_dir, '*.jpg'))
	dataset = np.array([scipy.misc.imread(path).astype(np.float) for path in paths])

# print(flags)
gan = DCGAN(width=flags.input_width, height=flags.input_height, channels=flags.channels)
gan.train(dataset)
gan.discriminator.save_weights('models/0{}_discrminator.h5'.format(flags.dataset))
gan.generator.save_weights('models/{0}_generator.h5'.format(flags.dataset))
gan.combined.save_weights('models/{0}_combined.h5'.format(flags.dataset))
