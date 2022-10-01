import os
from test_options import TestOptions
from data import create_dataset
from models import create_model
import torch

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    export_onnx_file = opt.name + ".onnx"
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, export_onnx_file)
    dummy_input = torch.randn(1, *(3, opt.load_size, opt.load_size)).cuda()
    torch.onnx.export(model.netG.module, dummy_input, save_path, opset_version=11, input_names = ['input'], output_names = ['output'])

    import onnx
    from onnxruntime_tools.quantization.quantize import quantize_dynamic, QuantType
    model_fp32 = save_path
    model_quant = save_path[:-5] + ".quant.onnx"
    quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
