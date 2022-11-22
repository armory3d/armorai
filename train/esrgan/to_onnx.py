
import torch
import torch.onnx
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
model.load_state_dict(torch.load('RealESRGAN_x2plus.pth')['params_ema']) # 'params'
model.train(False)
model.eval()

x = torch.rand(1, 3, 64, 64)
dynamic_axes= {'data':{2:'width', 3:'height'}, 'output':{2:'width', 3:'height'}}
with torch.no_grad():
    torch_out = torch.onnx._export(model, x, 'esrgan.onnx', input_names=['data'], output_names=['output'], dynamic_axes=dynamic_axes, opset_version=15)

from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
quantized_model = quantize_dynamic("esrgan.onnx", "esrgan.quant.onnx", weight_type=QuantType.QUInt8)
