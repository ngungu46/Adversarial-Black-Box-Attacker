from tensorflow.keras.models import load_model
import os
os.environ['TF_KERAS'] = '1'
import onnxmltools

# model_path = 'EfficientNetB0_butterfly.h5'
model_path = 'time.h5'

# out_path = 'Butterfly.onnx'
out_path = 'time.onnx'

model = load_model(model_path, custom_objects={'F1_score':'F1_score'})
onnx_model = onnxmltools.convert_keras(model) 

onnxmltools.utils.save_model(onnx_model, out_path)