
if __name__ == '__main__':
    from onnxruntime.quantization.quantize import quantize_dynamic, QuantType
    quantized_model = quantize_dynamic("CompVis/stable-diffusion-v1-4_onnx/unet/model.onnx", "sd_unet.onnx", weight_type=QuantType.QInt8)
