import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D

from sklearn.model_selection import train_test_split

lines = []

#Dataset with full track forward
with open('driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

for line in lines:
	source_path = line[0]
	filename = source_path.split('\\')[-1]
	current_path = 'IMG/' + filename
	line[0] = current_path
	source_path = line[1]
	filename = source_path.split('\\')[-1]
	current_path = 'IMG/' + filename
	line[1] = current_path
	source_path = line[2]
	filename = source_path.split('\\')[-1]
	current_path = 'IMG/' + filename
	line[2] = current_path

#Dataset with full track backward
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
	source_path = line1[1]
	filename = source_path.split('/')[-1]
	current_path = 'TrainingData1/IMG/' + filename
	line1[1] = current_path
	source_path = line1[2]
	filename = source_path.split('/')[-1]
	current_path = 'TrainingData1/IMG/' + filename
	line1[2] = current_path
	lines.append(line1)

#Dataset with only turns forward and backward
lines3 = []
with open('TrainingData3/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line3 in reader:
		lines3.append(line3)

for line3 in lines3:
	source_path = line3[0]
	filename = source_path.split('\\')[-1]
	current_path = 'TrainingData3/IMG/' + filename
	line3[0] = current_path
	source_path = line3[1]
	filename = source_path.split('\\')[-1]
	current_path = 'TrainingData3/IMG/' + filename
	line3[1] = current_path
	source_path = line3[2]
	filename = source_path.split('\\')[-1]
	current_path = 'TrainingData3/IMG/' + filename
	line3[2] = current_path
	lines.append(line3)

import random

#Save sample images
index = random.randint(0, len(lines))
test_line = lines[index]
test_center_image = cv2.imread(test_line[0])
cv2.imwrite('test_center_image.jpg', test_center_image)
test_left_image = cv2.imread(test_line[1])
cv2.imwrite('test_left_image.jpg', test_left_image)
test_right_image = cv2.imread(test_line[2])
cv2.imwrite('test_right_image.jpg', test_right_image)
test_flipped_image = np.fliplr(test_center_image)
cv2.imwrite('test_flipped_image.jpg', test_flipped_image)
test_cropped_image = test_center_image[50:140, 0:320]
cv2.imwrite('test_cropped_image.jpg', test_cropped_image)

#Split dataset into training and validation dataset
train_samples_temp, validation_samples_temp = train_test_split(lines, test_size=0.2)

train_angles = []
val_angles = []

import sys
from random import randint

#Remove some images with small angle to get good distribution
#for all angles
train_samples = []
total_count = 0
zero_count = 0
nonzero_count = 0
for sample in train_samples_temp:
	test = randint(0,9)
	if test == 3 or float(sample[3]) > 0.1 or float(sample[3]) < -0.1:
		train_samples.append(sample)
		total_count = total_count + 1
		if float(sample[3]) > 0.1 or float(sample[3]) < -0.1:
			total_count = total_count + 3
			nonzero_count = nonzero_count + 4
			train_samples.append(sample)
			train_samples.append(sample)
			train_samples.append(sample)
zero_count = total_count - nonzero_count
print("Total samples = ", total_count)
print("Straight drive training samples = ", zero_count)
print("Turn drive training samples = ", nonzero_count)

#Remove some images with small angle to get good distribution
#for all angles
validation_samples = []
total_count = 0
zero_count = 0
nonzero_count = 0
for sample in validation_samples_temp:
	test = randint(0,9)
	if test == 3 or float(sample[3]) > 0.1 or float(sample[3]) < -0.1:
		validation_samples.append(sample)
		total_count = total_count + 1
		if (float(sample[3]) > 0.1 or float(sample[3]) < -0.1):
			total_count = total_count + 3
			nonzero_count = nonzero_count + 4
			validation_samples.append(sample)
			validation_samples.append(sample)
			validation_samples.append(sample)
zero_count = total_count - nonzero_count
print("Total samples = ", total_count)
print("Straight drive training samples = ", zero_count)
print("Turn drive training samples = ", nonzero_count)

print("Visualize dataset")
for sample in train_samples:
	train_angles.append(sample[3])
for sample in validation_samples:
	val_angles.append(sample[3])

n_train_angles = np.unique(train_angles).shape[0]
n_val_angles = np.unique(val_angles).shape[0]

print("Number of unique angles in training dataset =", n_train_angles)
print("Number of unique angles in validation dataset =", n_val_angles)

#Plot histogram for training data
fig, ax = plt.subplots()
ind, n_bins = np.histogram(np.array(train_angles).astype(np.float), bins=n_train_angles)
n_bins = np.delete(n_bins, len(n_bins)-1)
rect = ax.bar(n_bins, ind)
ax.set_ylabel('Count')
ax.set_title('training data')
fig.savefig('training_data.png')

#Plot histogram for validation data
fig, ax = plt.subplots()
ind, n_bins = np.histogram(np.array(val_angles).astype(np.float), bins=n_val_angles)
n_bins = np.delete(n_bins, len(n_bins)-1)
rect = ax.bar(n_bins, ind)
ax.set_ylabel('Count')
ax.set_title('validation data')
fig.savefig('validation_data.png')

#Implement generator
def generator(samples, batch_size=16, training=True):
	num_samples = len(samples)

	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []

			for batch_sample in batch_samples:
				#Images from center camera
				name = batch_sample[0]
				center_image = cv2.imread(name)
				center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)

				#Augment data only for training and if angle is not zero
				if (training == True and center_angle != 0):
					#Flip images from center camera
					image_flipped = np.fliplr(center_image)
					angle_flipped = -center_angle
					images.append(image_flipped)
					angles.append(angle_flipped)

					#Images from left camera
					left_name = batch_sample[1]
					left_image = cv2.imread(left_name)
					left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
					left_angle = center_angle + 0.2
					images.append(left_image)
					angles.append(left_angle)

					#Images from right camera
					right_name = batch_sample[2]
					right_image = cv2.imread(right_name)
					right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
					right_angle = center_angle - 0.2
					images.append(right_image)
					angles.append(right_angle)

			X_train = np.array(images)
			y_train = np.array(angles)

			yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
	Model = Sequential()
	Model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320,3)))
	Model.add(Lambda(lambda x: x/127.5 - 1.0))
	Model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), border_mode="same", use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Flatten())
	Model.add(Dense(100, use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Dropout(.5))
	Model.add(Dense(50, use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Dropout(.5))
	Model.add(Dense(10, use_bias=True, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
	Model.add(Dense(1))
	Model.compile(loss='mse', optimizer='adam')

	return Model

train_generator = generator(train_samples, batch_size=16, training=True)
validation_generator = generator(validation_samples, batch_size=16, training=False)

if __name__ == '__main__':
	print("Number of samples {0}".format(len(train_samples)))
	print("Number of batches {0}".format((len(train_samples) + 15)/16))
	model = get_model()
	model.fit_generator(train_generator, steps_per_epoch=(len(train_samples)+15)/16,\
			validation_data=validation_generator, nb_val_samples=len(validation_samples),\
			nb_epoch=10)
	model.save('model.h5')
