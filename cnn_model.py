#!/usr/bin/env python3

#created by jcsendin on the 1/April/2022


#INFORMATION-------------------------------
#In this file we can find all the code regarding the creation and training of our cnn


import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt


#DATA--------------------------------------

#Metadata
EPOCHS = 1
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


#FUNCTIONS---------------------------------

#this function get the train and test sets normalized
def data_processing(dataset):
	(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

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
def build_model(input_shape, output_shape):
	model = tfk.models.Sequential()
	model.add(tfk.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
	model.add(tfk.layers.MaxPooling2D((2, 2)))

	model.add(tfk.layers.Flatten())
	model.add(tfk.layers.Dense(64, activation='relu'))
	model.add(tfk.layers.Dropout(0.3))
	model.add(tfk.layers.Dense(output_shape, activation='softmax'))

	model.compile(optimizer='adam',
					loss=tfk.losses.SparseCategoricalCrossentropy(),
					metrics=['accuracy'])

	model.summary()

	return model


def train_model(model, train_data, val_data, epochs, batch_size, es_patience):
	return model.fit(train_data[0], 
						train_data[1],
						epochs=epochs, 
						validation_data=val_data,
						batch_size = batch_size,
						callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy',
															mode='max',
															patience=es_patience,
															restore_best_weights=True)])


def show_metrics(history):
	plt.plot(history.history['accuracy'], label='accuracy')
	plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.5, 1])
	plt.legend(loc='lower right')
	

#this function creates and trains the model returning it. It also shows some metrics of the trained model
def prepare_model(train_data, test_data, input_shape, output_shape, epochs, batch_size,class_names=[]):
	model = build_model(input_shape, output_shape)
	history = train_model(model, train_data, test_data, epochs, batch_size, 5)

	show_metrics(history)

	return model


'''
if __name__ == "__main__":
	#working with the code
	train_data, test_data = data_processing(DATASET)
	model = prepare_model(train_data, test_data, INPUT_SHAPE, OUTPUT_SHAPE, EPOCHS, BATCH_SIZE)
	test_loss, test_acc = model.evaluate(test_data[0],  test_data[1], verbose=2)
	print(test_acc)
'''