#!/usr/bin/env python3

#created by jcsendin on the 2/April/2022


#INFORMATION-------------------------------
#In this file we can find all the code regarding the creation of the Fast Gradient Sign Method (FGSM)


#IMPORTS-----------------------------------
import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.io
import numpy as np

from cnn_model import LOSS_OBJECT


#FUNCTIONS---------------------------------

def preprocess(image):
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, (32, 32))
	image = image/255.0 #Normalization
	image = image[None,...]

	return image


def get_labels(image_probs, class_names):
	pred_pos = np.argmax(image_probs)
	image_class = class_names[pred_pos]
	class_confidence = image_probs[pred_pos]

	return image_class, class_confidence


def display_image(image, description, label, confidence):
	plt.figure()
	plt.imshow(image[0]*0.5+0.5)
	plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
													label, confidence*100))
	plt.show()


def create_adversarial_pattern(model, input_image, input_label):
	with tf.GradientTape() as tape:
		tape.watch(input_image)
		prediction = model(input_image)
		loss = LOSS_OBJECT(input_label, prediction)

	gradient = tape.gradient(loss, input_image)
	signed_grad = tf.sign(gradient)

	return signed_grad


def create_adversarial_image(model, image, label, eps=0.1):
	perturbations = create_adversarial_pattern(model, image, label)
	adversary = image + eps*perturbations
	adversary = tf.clip_by_value(adversary, -1, 1)

	return adversary
	