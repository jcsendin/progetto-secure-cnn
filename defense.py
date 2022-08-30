#!/usr/bin/env python3

#created by jcsendin on the 3/April/2022


#INFORMATION-------------------------------
#In this file we can find two defensive approaches against a FGSM attack


#IMPORTS-----------------------------------
import tensorflow as tf
import numpy as np

from fgsm import create_adversarial_image
from sklearn.utils import shuffle

from constants import *


#FUNCTIONS---------------------------------

def generate_adversarial_batch(model, total, images, labels, dims, eps=EPSILON):
	while True:
		perturbImages = []
		perturbLabels = []

		idxs = np.random.choice(range(0, len(images)), size = total, replace = False)

		for i in idxs:
			image = images[i]
			image = tf.convert_to_tensor(image)
			image = image[None, ...]
			label = labels[i]

			adversary = create_adversarial_image(model, image, label, eps)

			perturbImages.append(tf.reshape(adversary, dims))
			perturbLabels.append(label)

		yield (np.array(perturbImages), np.array(perturbLabels))


def generate_mixed_adversarial_batch(model, total, images, labels, dims, eps=EPSILON, split=SPLIT):
	#the split indicates the percentage of adversarial images
	totalAdv = int(total * split)
	totalNormal = int(total * (1 - split))

	while True:
		idxs = np.random.choice(range(0, len(images)),
			size=totalNormal, replace=False)
		mixedImages = images[idxs]
		mixedLabels = labels[idxs]
		
		idxs = np.random.choice(range(0, len(images)), size=totalAdv,
			replace=False)
  
		for i in idxs:
			image = images[i]
			image = tf.convert_to_tensor(image)
			image = image[None, ...]
			label = labels[i]

			adversary = create_adversarial_image(model, image, label, eps)

			mixedImages = np.vstack([mixedImages, adversary])
			mixedLabels = np.vstack([mixedLabels, label])

		(mixedImages, mixedLabels) = shuffle(mixedImages, mixedLabels)

		yield (mixedImages, mixedLabels)
