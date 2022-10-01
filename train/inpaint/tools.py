import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)

def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including img
    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64)

def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.size(0)
    mask = torch.zeros((batch_size, 1, height, width), dtype=torch.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.0
    return mask

def mask_image(x, bboxes, config):
    height, width, _ = config['image_shape']
    max_delta_h, max_delta_w = config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    if x.is_cuda:
        mask = mask.cuda()
    result = x * (1.0 - mask)
    return result, mask

def get_config():
    config = {}
    config['dataset_name'] = 'photo_inpaint'
    config['train_data_path'] = 'C:/dev/armorai/datasets/photo_inpaint/'
    config['resume'] = None
    config['batch_size'] = 16
    config['image_shape'] = [256, 256, 3]
    config['mask_shape'] = [128, 128]
    config['mask_batch_same'] = True
    config['max_delta_shape'] = [32, 32]
    config['margin'] = [0, 0]
    config['discounted_mask'] = True
    config['spatial_discounting_gamma'] = 0.9
    config['random_crop'] = True
    config['mosaic_unit_size'] = 12
    config['cuda'] = True
    config['gpu_ids'] = [0]
    config['lr'] = 0.0001
    config['beta1'] = 0.5
    config['beta2'] = 0.9
    config['n_critic'] = 5
    config['niter'] = 2000000
    config['print_iter'] = 100
    config['viz_iter'] = 1000
    config['viz_max_out'] = 16
    config['snapshot_save_iter'] = 5000
    config['coarse_l1_alpha'] = 1.2
    config['l1_loss_alpha'] = 1.2
    config['ae_loss_alpha'] = 1.2
    config['global_wgan_loss_alpha'] = 1.0
    config['gan_loss_alpha'] = 0.001
    config['wgan_gp_lambda'] = 10
    config['netG'] = {}
    config['netG']['input_dim'] = 3
    config['netG']['ngf'] = 32
    config['netD'] = {}
    config['netD']['input_dim'] = 3
    config['netD']['ndf'] = 64
    return config

def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name
