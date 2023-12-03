import onnx
from onnx2pytorch import ConvertModel

# Path to ONNX model
# onnx_model_path = 'Butterfly.onnx'
onnx_model_path = 'time.onnx'

# Or you can load a regular onnx model and pass it to the converter
onnx_model = onnx.load(onnx_model_path)
pytorch_model = ConvertModel(onnx_model)
