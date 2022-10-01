import os
import random
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch import autograd
from tools import get_config, random_bbox, mask_image, default_loader, normalize, get_model_list
from os import listdir
from networks import Generator, LocalDis, GlobalDis

def local_patch(x, bbox_list):
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)

def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['spatial_discounting_gamma']
    height, width = config['mask_shape']
    shape = [1, 1, height, width]
    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch.float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor

def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

class Dataset(data.Dataset):
    def __init__(self, data_path, image_shape, random_crop=True):
        super(Dataset, self).__init__()
        self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop

    def __getitem__(self, index):
        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)
        img = normalize(img)
        return img

    def __len__(self):
        return len(self.samples)

class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2 = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)

        # D part
        # wgan d loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * self.config['coarse_l1_alpha'] + l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
            losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * self.config['coarse_l1_alpha'] + l1_loss(x2 * (1.0 - masks), ground_truth * (1.0 - masks))

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self.localD, local_patch_gt, local_patch_x2_inpaint)
            global_real_pred, global_fake_pred = self.dis_forward(self.globalD, ground_truth, x2_inpaint)
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x2_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2 = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1.0 - masks)
        x2_inpaint = x2 * masks + x * (1.0 - masks)

        return x2_inpaint

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration

def main():
    config = get_config()

    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    checkpoint_path = os.path.join('checkpoints', config['dataset_name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    args_seed = random.randint(1, 10000)
    random.seed(args_seed)
    torch.manual_seed(args_seed)
    if cuda:
        torch.cuda.manual_seed_all(args_seed)

    train_dataset = Dataset(data_path=config['train_data_path'],
                            image_shape=config['image_shape'],
                            random_crop=config['random_crop'])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=1)

    trainer = Trainer(config)

    if cuda:
        trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
        trainer_module = trainer.module
    else:
        trainer_module = trainer

    start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

    iterable_train_loader = iter(train_loader)

    time_count = time.time()

    for iteration in range(start_iteration, config['niter'] + 1):
        try:
            ground_truth = next(iterable_train_loader)
        except StopIteration:
            iterable_train_loader = iter(train_loader)
            ground_truth = next(iterable_train_loader)

        # Prepare the inputs
        bboxes = random_bbox(config, batch_size=ground_truth.size(0))
        x, mask = mask_image(ground_truth, bboxes, config)
        if cuda:
            x = x.cuda()
            mask = mask.cuda()
            ground_truth = ground_truth.cuda()

        ###### Forward pass ######
        compute_g_loss = iteration % config['n_critic'] == 0
        losses, inpainted_result = trainer(x, bboxes, mask, ground_truth, compute_g_loss)
        # Scalars from different devices are gathered into vectors
        for k in losses.keys():
            if not losses[k].dim() == 0:
                losses[k] = torch.mean(losses[k])

        ###### Backward pass ######
        # Update D
        trainer_module.optimizer_d.zero_grad()
        losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
        losses['d'].backward()
        trainer_module.optimizer_d.step()

        # Update G
        if compute_g_loss:
            trainer_module.optimizer_g.zero_grad()
            losses['g'] = losses['l1'] * config['l1_loss_alpha'] + losses['ae'] * config['ae_loss_alpha'] + losses['wgan_g'] * config['gan_loss_alpha']
            losses['g'].backward()
            trainer_module.optimizer_g.step()

        # Log and visualization
        log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
        if iteration % config['print_iter'] == 0:
            time_count = time.time() - time_count
            speed = config['print_iter'] / time_count
            speed_msg = 'speed: %.2f batches/s ' % speed
            time_count = time.time()

            message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
            for k in log_losses:
                v = losses.get(k, 0.)
                message += '%s: %.6f ' % (k, v)
            message += speed_msg
            print(message)

        if iteration % (config['viz_iter']) == 0:
            viz_max_out = config['viz_max_out']
            if x.size(0) > viz_max_out:
                viz_images = torch.stack([x[:viz_max_out], inpainted_result[:viz_max_out]], dim=1)
            else:
                viz_images = torch.stack([x, inpainted_result], dim=1)
            viz_images = viz_images.view(-1, *list(x.size())[1:])
            vutils.save_image(viz_images, '%s/niter_%03d.png' % (checkpoint_path, iteration), nrow=3 * 4, normalize=True)

        # Save the model
        if iteration % config['snapshot_save_iter'] == 0:
            trainer_module.save_model(checkpoint_path, iteration)

if __name__ == '__main__':
    main()
