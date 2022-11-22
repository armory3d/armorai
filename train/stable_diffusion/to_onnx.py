
from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
quantized_model = quantize_dynamic("stable_diffusion_onnx/unet/model.onnx", "sd_unet.quant.onnx", weight_type=QuantType.QUInt8)
quantized_model = quantize_dynamic("stable_diffusion_onnx/text_encoder/model.onnx", "sd_text_encoder.quant.onnx", weight_type=QuantType.QUInt8)
quantized_model = quantize_dynamic("stable_diffusion_onnx/vae_decoder/model.onnx", "sd_vae_decoder.quant.onnx", weight_type=QuantType.QUInt8)
quantized_model = quantize_dynamic("stable_diffusion_onnx/vae_encoder/model.onnx", "sd_vae_encoder.quant.onnx", weight_type=QuantType.QUInt8)
