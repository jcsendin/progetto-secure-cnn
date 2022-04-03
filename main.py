#!/usr/bin/env python3

#created by jcsendin on the 2/April/2022

#This is the "main" file where we execute our code

#IMPORTS-----------------------------------
from cnn_model import *
from fgsm import *


if __name__ == "__main__":

	#First step - Creating our model and training it
	train_data, test_data = data_processing()
	model = prepare_model(train_data, test_data)
	test_loss, test_acc = model.evaluate(test_data[0],	test_data[1], verbose=2)
	print(test_acc)

	#Second step - Creating adversarial images
	image_array = skimage.io.imread('https://upload.wikimedia.org/wikipedia/commons/2/24/Stray_calico_cat_near_Sagami_River-01.jpg')
	sample_label = CLASS_NAMES.index("cat")

	image = preprocess(image_array)	
	image_probs = model.predict(image)

	image_class, confidence = get_labels(image_probs[0], CLASS_NAMES)

	perturbations = create_adversarial_pattern(model, image, sample_label)
	epsilons = [0, 0.00005, 0.0001, 0.0005, 0.001, 0.005]
	descriptions = [('Epsilon = {:0.5f}'.format(eps) if eps else 'Input')
				for eps in epsilons]

	for i, eps in enumerate(epsilons):
		adv_x = image + eps*perturbations
		adv_x = tf.clip_by_value(adv_x, -1, 1)

		image_probs = model.predict(image)
		image_class, confidence = get_labels(image_probs[0], CLASS_NAMES)

		display_image(adv_x, descriptions[i],image_class, confidence)