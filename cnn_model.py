#!/usr/bin/env python3

#created by jcsendin on the 1/April/2022


#INFORMATION-------------------------------
#In this file we can find all the code regarding the creation and training of our cnn


#IMPORTS-----------------------------------
import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt


#DATA--------------------------------------

#Metadata
EPOCHS = 200
BATCH_SIZE = 64

#Dataset we will be using - Cifar10
DATASET = tfk.datasets.cifar10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
				 'dog', 'frog', 'horse', 'ship', 'truck']
N_CLASSES = 10
INPUT_SHAPE = (32,32,3)
OUTPUT_SHAPE = 10

IMG_WIDTH= 255
IMG_HEIGHT = 255
IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT)

LOSS_OBJECT = tf.keras.losses.SparseCategoricalCrossentropy()


#FUNCTIONS---------------------------------

#this function get the train and test sets normalized
def data_processing():
	(train_images, train_labels), (test_images, test_labels) = DATASET.load_data()

	# Normalize pixel values to be between 0 and 1 and return 
	return (train_images / 255.0, train_labels), (test_images / 255.0, test_labels)


def show_classes(data, classes):
	plt.figure(figsize=(len(classes),len(classes)))
	for i in range(25):
		plt.subplot(5,5,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(data[0][i])
		plt.xlabel(classes[data[1][i][0]])
	plt.show()


#this is an example model, it is not the best model but works fine for what we want to achieve
def build_model():
	model = tfk.models.Sequential()
	model.add(tfk.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE, padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Flatten())
	model.add(tfk.layers.Dense(64, activation='relu'))
	model.add(tfk.layers.Dropout(0.3))
	model.add(tfk.layers.Dense(OUTPUT_SHAPE, activation='softmax'))

	model.compile(optimizer='adam',
					loss=LOSS_OBJECT,
					metrics=['accuracy'])

	model.summary()

	return model


def train_model(model, train_data, val_data):
	return model.fit(train_data[0], 
						train_data[1],
						epochs=EPOCHS, 
						validation_data=val_data,
						batch_size = BATCH_SIZE,
						callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy',
															mode='max',
															patience=5,
															restore_best_weights=True)])


def show_metrics(history):
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')

	plt.show()
	

#this function creates and trains the model returning it. It also shows some metrics of the trained model
def prepare_model(train_data, test_data):
	model = build_model()
	history = train_model(model, train_data, test_data)

	show_metrics(history)

	return model
