import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.onnx
from networks import Generator
from tools import get_config, random_bbox, mask_image, default_loader, normalize, get_model_list

def main():
    args = {}
    args['image'] = './examples/b.jpg'
    args['mask'] = './examples/center_mask_256.png'
    args['output'] = './examples/output.png'
    args['iter'] = 0
    config = get_config()

    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    args_seed = random.randint(1, 10000)
    random.seed(args_seed)
    torch.manual_seed(args_seed)
    if cuda:
        torch.cuda.manual_seed_all(args_seed)

    with torch.no_grad():   # enter no grad context
        if args['mask']:
            # Test a single masked image with a given mask
            x = default_loader(args['image'])
            mask = default_loader(args['mask'])
            x = transforms.Resize(config['image_shape'][:-1])(x)
            x = transforms.CenterCrop(config['image_shape'][:-1])(x)
            mask = transforms.Resize(config['image_shape'][:-1])(mask)
            mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
            x = transforms.ToTensor()(x)
            mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
            x = normalize(x)
            x = x * (1. - mask)
            x = x.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)
        else:
            # Test a single ground-truth image with a random mask
            ground_truth = default_loader(args['image'])
            ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
            ground_truth = transforms.CenterCrop(config['image_shape'][:-1])(ground_truth)
            ground_truth = transforms.ToTensor()(ground_truth)
            ground_truth = normalize(ground_truth)
            ground_truth = ground_truth.unsqueeze(dim=0)
            bboxes = random_bbox(config, batch_size=ground_truth.size(0))
            x, mask = mask_image(ground_truth, bboxes, config)

        checkpoint_path = os.path.join('checkpoints', config['dataset_name'])

        # Define the trainer
        netG = Generator(config['netG'], cuda, device_ids)
        # Resume weight
        last_model_name = get_model_list(checkpoint_path, "gen", iteration=args['iter'])
        netG.load_state_dict(torch.load(last_model_name))

        model_iteration = int(last_model_name[-11:-3])
        print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

        if cuda:
            netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
            x = x.cuda()
            mask = mask.cuda()

        # Inference
        # x1, x2 = netG(x, mask)
        x2 = netG(x, mask)
        inpainted_result = x2 * mask + x * (1.0 - mask)

        vutils.save_image(inpainted_result, args['output'], padding=0, normalize=True)
        print("Saved the inpainted result to {}".format(args['output']))

        ####

        export_onnx_file = "photo_inpaint.onnx"
        save_dir = "checkpoints"
        save_path = os.path.join(save_dir, export_onnx_file)
        dummy_input = torch.randn(1, *(3, config['image_shape'][0], config['image_shape'][1]))
        dummy_mask = torch.randn(1, *(1, config['image_shape'][0], config['image_shape'][1]))
        if cuda:
            dummy_input = dummy_input.cuda()
            dummy_mask = dummy_mask.cuda()
            netG = netG.module

        torch.onnx.export(netG, (dummy_input, dummy_mask), save_path, opset_version=11, input_names=['input', 'input_mask'], output_names=['output'])

        # import onnx
        # from onnx import helper, shape_inference
        # onnx.checker.check_model(original_model)
        # inferred_model = onnx.shape_inference.infer_shapes(original_model)
        # onnx.checker.check_model(inferred_model)

        # import onnx
        # from onnxruntime_tools.quantization.quantize import quantize_dynamic, QuantType
        # model_fp32 = save_path
        # model_quant = save_path[:-5] + ".quant.onnx"
        # quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

if __name__ == '__main__':
    main()
