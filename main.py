#!/usr/bin/env python3

#created by jcsendin on the 2/April/2022

#INFORMATION-------------------------------
#This is the "main" file where we execute our code

#IMPORTS-----------------------------------
import tensorflow.keras as tfk
import os

from cnn_model import *
from fgsm import *
from defense import *


if __name__ == "__main__":

	#FIRST STEP - Creating our model and training it
	train_data, test_data = data_processing()

	train_images, train_labels = train_data
	test_images, test_labels = test_data

	model = prepare_model(train_data, test_data)
	test_loss, test_acc = model.evaluate(test_images,	test_labels, verbose=2)
	print(test_acc)

	#Saving the model in ONNX format
	model.save("savedmodel")
	os.system("python -m tf2onnx.convert --saved-model savedmodel --output models/BaseModel.onnx")


	#SECOND STEP - Creating adversarial images
	#Uploading an image of a cat
	image_array = skimage.io.imread('https://upload.wikimedia.org/wikipedia/commons/2/24/Stray_calico_cat_near_Sagami_River-01.jpg')
	sample_label = CLASS_NAMES.index("cat")

	image = preprocess(image_array)	
	image_probs = model.predict(image)

	#Getting the confidence of the prediction
	image_class, confidence = get_labels(image_probs[0], CLASS_NAMES)

	perturbations = create_adversarial_pattern(model, image, sample_label)
	epsilons = [0.001, 0.005, 0.01, 0.05, 0.1]
	descriptions = [('Epsilon = {:0.5f}'.format(eps) if eps else 'Input')
				for eps in epsilons]

	#Playing with different epsilons values to see the change in confidence
	for i, eps in enumerate(epsilons):
		adv_x = image + eps*perturbations
		adv_x = tf.clip_by_value(adv_x, -1, 1)

		image_probs = model.predict(image)
		image_class, confidence = get_labels(image_probs[0], CLASS_NAMES)

		#display_image(adv_x, descriptions[i],image_class, confidence)
		

	#THIRD STEP - Applying some defenses

	#Showing the effect of the adversarial attack 
	adv_test_images, adv_test_labels = next(generate_adversarial_batch(model, len(test_images), test_images, test_labels, INPUT_SHAPE, 0.1) )
	adv_loss, adv_acc = model.evaluate(x=adv_test_images, y=adv_test_labels, verbose=0)

	print(adv_acc)


	#We are going to start with a first defense. We will fine-tune our model with a batch of adversarial images
	adv_train_images, adv_train_labels = next(generate_adversarial_batch(model, len(train_images), train_images, train_labels, (32, 32, 3), 0.1) )

	#To fine tune we decrease the learning rate of the optimizer
	model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
			loss=LOSS_OBJECT,
			metrics=['accuracy'])

	model.fit(adv_train_images, adv_train_labels, 
		 	validation_data = (adv_test_images, adv_test_labels),
		  	epochs=EPOCHS, callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)])

	#Comparing the accuracy between a normal batch and an adversarial batch
	adv_loss, adv_acc = model.evaluate(x=adv_test_images, y=adv_test_labels, verbose=0)
	print("Adversarial Test Images Accuracy",adv_acc)

	adv_loss, adv_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)
	print("Test Images Accuracy",adv_acc)

	#Saving the model in ONNX format
	model.save("savedmodel")
	os.system("python -m tf2onnx.convert --saved-model savedmodel --output models/ModelFirstDef.onnx")


	#Second method, we are going to fine-tune with a mixed batch of normal and adversarial images
	train_mix_images, train_mix_labels = next(generate_mixed_adversarial_batch(model, len(train_images), train_images, train_labels, (32, 32, 1), eps=0.1, split=0.1))
	test_mix_images, test_mix_labels = next(generate_mixed_adversarial_batch(model, len(test_images), test_images, test_labels, (32, 32, 1), eps=0.1, split=0.1))

	model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
			loss=LOSS_OBJECT,
			metrics=['accuracy'])

	model.fit(train_mix_images,
			train_mix_labels,
			validation_data = (test_mix_images, test_mix_labels),
			batch_size = BATCH_SIZE, 
			steps_per_epoch = len(train_images) // BATCH_SIZE, 
			epochs=EPOCHS,
			callbacks = [tfk.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)])

	#Testing the accuracy
	adv_loss, adv_acc = model.evaluate(x=test_images, y=test_labels, verbose=0)
	print("Test Images Accuracy",adv_acc)

	adv_loss, adv_acc = model.evaluate(x=adv_test_images, y=adv_test_labels, verbose=0)
	print("Adversarial Test Images Accuracy",adv_acc)

	#Saving the model in ONNX format
	model.save("savedmodel")
	os.system("python -m tf2onnx.convert --saved-model savedmodel --output models/ModelSecondDef.onnx")