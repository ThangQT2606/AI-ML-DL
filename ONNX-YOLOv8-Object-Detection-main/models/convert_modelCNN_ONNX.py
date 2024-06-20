import onnx
import os
import tf2onnx #keras2onnx, 
import tensorflow as tf

model_path = os.path.join(os.path.dirname(__file__), 'Best_model.h5')
#Load The saved model 
from keras.models import load_model
loaded_keras_model = load_model(model_path)

onnx_model, _ = tf2onnx.convert.from_keras(loaded_keras_model)

onnx.save(onnx_model, 'classifi.onnx')