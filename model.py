#import libraries
import numpy as np
import os, csv 
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Convolution2D, Cropping2D, Dropout
from keras.preprocessing.image import random_shear
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Parse the driving log file
samples = []
log_path = './data/all/driving_log.csv'
with open(log_path) as csvlogfile:
	reader_csv = csv.reader(csvlogfile)
	for line in reader_csv:
		samples.append(line)

# Create train set and validation set(20% of the data)
train_set, validation_set = train_test_split(samples, test_size=0.2)

# Dimesions of the picture:
cols = 320
rows = 160
channels = 3 

# Add and offset for the left and right camera
angle_offsets = [0, 0.23, -0.23]

# Python generator to generate data for training on the fly, rather than storing everything in memory
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:

				for i, correction in zip(range(3), angle_offsets):

					name = './data/all/IMG/' + batch_sample[i].split('/')[-1]
					#print (name)
					image = mpimg.imread(name)
					angle = float(batch_sample[3]) + correction

					# original image
					images.append(image)
					angles.append(angle)

					# flip image (invert angle)
					images.append(np.fliplr(image))
					angles.append(angle * -1.0)

					# random shear
					images.append(random_shear(image, np.random.randint(32)))
					angles.append(angle)

			# convert to np array
			X_train = np.array(images)
			y_train = np.array(angles)
			#Shuffle the data
			yield shuffle(X_train, y_train)

# Define batch size and generators
batch_size = 256
train_generator = generator(train_set, batch_size=batch_size)
validation_generator = generator(validation_set, batch_size=batch_size)

# create model
model = Sequential()
# normalize the data and traslate to have zero mean
model.add(Lambda(lambda x: x/(255/2) - 1.0, input_shape=(rows,cols,channels)))
# define layers
model.add(Cropping2D(cropping=((60,20), (0,0))))
model.add(Convolution2D(16,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(32,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(200))
model.add(Dropout(0.7))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(1))

# compile the model
model.compile(loss='mse', optimizer='adam')

# Train
nb_epochs = 3
model.fit_generator(train_generator, samples_per_epoch=len(train_set)*10, validation_data=validation_generator, nb_val_samples=len(validation_set), nb_epoch=nb_epochs)

# save the model
model.save('model.h5')
