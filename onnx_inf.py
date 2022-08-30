#!/usr/bin/env python3

#created by jcsendin on the 30/August/2022


#INFORMATION-------------------------------
#In this file we can find all the code regarding the little ONNX inference experiments we are going to perform


#IMPORTS-----------------------------------
import time
import onnx
import onnxruntime as ort
import numpy as np

from constants import *


#FUNCTIONS-----------------------------------

def measure_time(fct, imgs, n, timeout=30):
	times = []
	for i in range(0, n):
		img = imgs[i % len(imgs)]
		aux = np.expand_dims(img, axis=0)
		aux = aux.astype(np.float32)

		begin = time.perf_counter()
		result = fct(aux)
		end = time.perf_counter()

		times.append(end - begin)
		if sum(times) > timeout:
			break
			
	return times


def execute_ONNX_model_N_times(model, images, N):
	onnx_model = onnx.load(model)
	onnx_model_str = onnx_model.SerializeToString()
	providers = ort.get_available_providers()
	for provider in providers:
		options = ort.SessionOptions()
		options.enable_profiling = True
		sess_profile = ort.InferenceSession(onnx_model_str, options, providers=providers)
		input_name = sess_profile.get_inputs()[0].name
		fct = lambda img: sess_profile.run(None, {input_name:img})
		times = measure_time(fct, images, n=N)
		prof_file = sess_profile.end_profiling()
		return [sum(times)]
	

def execute_model_N_batch(model, images, N):
	begin = time.perf_counter()
	model.predict(images[0:N])
	end = time.perf_counter()
	return end-begin