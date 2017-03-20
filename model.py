import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D

from sklearn.model_selection import train_test_split

import sklearn

lines = []
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

for line in lines:
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = 'IMG/' + filename
	line[0] = current_path

lines1 = []
with open('TrainingData1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line1 in reader:
		lines1.append(line1)

for line1 in lines1:
	source_path = line1[0]
	filename = source_path.split('/')[-1]
	current_path = 'TrainingData1/IMG/' + filename
	line1[0] = current_path
	lines.append(line1)

lines2 = []
with open('SampleData/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line2 in reader:
		lines2.append(line2)

for line2 in lines2:
	source_path = line2[0]
	filename = source_path.split('/')[-1]
	current_path = 'SampleData/IMG/' + filename
	line2[0] = current_path
	lines.append(line2)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=16):
	num_samples = len(samples)

	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []

			for batch_sample in batch_samples:
				name = batch_sample[0]
				center_image = cv2.imread(name)
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
	Model = Sequential()
	Model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320,3)))
	Model.add(Lambda(lambda x: x/127.5 - 1.0))
	Model.add(Convolution2D(filters=16, kernel_size=(4, 4), strides=(4, 4), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Convolution2D(filters=32, kernel_size=(3, 3), strides=(2, 2), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform',bias_initializer='zeros'))
	Model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Flatten())
	Model.add(Dropout(.5))
	Model.add(ELU())
	Model.add(Dense(512))
	Model.add(Dropout(.5))
	Model.add(ELU())
	Model.add(Dense(1))
	Model.compile(loss='mse', optimizer='adam')

	return Model

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

if __name__ == '__main__':
	print("Number of samples {0}".format(len(train_samples)))
	print("Number of batches {0}".format((len(train_samples) + 15)/16))
	model = get_model()
	model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)+15)/16,\
			validation_data=validation_generator, nb_val_samples=len(validation_samples),\
			nb_epoch=5)
	model.save('model.h5')
