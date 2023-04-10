
**Setup**
```bash
conda env create -f environment.yml
conda activate pix2pix
```

**Photo to Base**
```bash
python combine_A_and_B.py --fold_A datasets/photo_to_base/A --fold_B datasets/photo_to_base/B --fold_AB datasets/photo_to_base --no_multiprocessing
python train.py --dataroot ./datasets/photo_to_base --name photo_to_base --model pix2pix --direction BtoA --load_size 2176 --crop_size 2176 --save_epoch_freq 50 --netG unet_128 --n_epochs 200 --n_epochs_decay 200 --norm instance --display_id 0
python to_onnx.py --dataroot ./datasets/samples --name photo_to_base --model test --direction BtoA --load_size 2176 --crop_size 2176 --netG unet_128 --dataset_mode single --no_dropout --norm instance
```

**Photo to Normal**
```bash
python combine_A_and_B.py --fold_A datasets/photo_to_normal/A --fold_B datasets/photo_to_normal/B --fold_AB datasets/photo_to_normal --no_multiprocessing
python train.py --dataroot ./datasets/photo_to_normal --name photo_to_normal --model pix2pix --direction BtoA --load_size 2176 --crop_size 2176 --save_epoch_freq 50 --netG unet_128 --n_epochs 200 --n_epochs_decay 200 --norm instance --display_id 0
python to_onnx.py --dataroot ./datasets/samples --name photo_to_normal --model test --direction BtoA --load_size 2176 --crop_size 2176 --netG unet_128 --dataset_mode single --no_dropout --norm instance
```

**Photo to Height**
```bash
python combine_A_and_B.py --fold_A datasets/photo_to_height/A --fold_B datasets/photo_to_height/B --fold_AB datasets/photo_to_height --no_multiprocessing
python train.py --dataroot ./datasets/photo_to_height --name photo_to_height --model pix2pix --direction BtoA --load_size 2176 --crop_size 2176 --save_epoch_freq 50 --netG unet_128 --n_epochs 200 --n_epochs_decay 200 --norm instance --display_id 0 --output_nc 1
python to_onnx.py --dataroot ./datasets/samples --name photo_to_height --model test --direction BtoA --load_size 2176 --crop_size 2176 --netG unet_128 --dataset_mode single --no_dropout --norm instance --output_nc 1
```

**Photo to Occlusion**
```bash
python combine_A_and_B.py --fold_A datasets/photo_to_occlusion/A --fold_B datasets/photo_to_occlusion/B --fold_AB datasets/photo_to_occlusion --no_multiprocessing
python train.py --dataroot ./datasets/photo_to_occlusion --name photo_to_occlusion --model pix2pix --direction BtoA --load_size 2176 --crop_size 2176 --save_epoch_freq 50 --netG unet_128 --n_epochs 200 --n_epochs_decay 200 --norm instance --display_id 0 --output_nc 1
python to_onnx.py --dataroot ./datasets/samples --name photo_to_occlusion --model test --direction BtoA --load_size 2176 --crop_size 2176 --netG unet_128 --dataset_mode single --no_dropout --norm instance --output_nc 1
```

**Photo to Roughness**
```bash
python combine_A_and_B.py --fold_A datasets/photo_to_roughness/A --fold_B datasets/photo_to_roughness/B --fold_AB datasets/photo_to_roughness --no_multiprocessing
python train.py --dataroot ./datasets/photo_to_roughness --name photo_to_roughness --model pix2pix --direction BtoA --load_size 2176 --crop_size 2176 --save_epoch_freq 50 --netG unet_128 --n_epochs 200 --n_epochs_decay 200 --norm instance --display_id 0 --output_nc 1
python to_onnx.py --dataroot ./datasets/samples --name photo_to_roughness --model test --direction BtoA --load_size 2176 --crop_size 2176 --netG unet_128 --dataset_mode single --no_dropout --norm instance --output_nc 1
```
