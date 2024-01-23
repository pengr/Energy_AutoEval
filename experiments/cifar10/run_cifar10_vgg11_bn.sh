#!/bin/bash
cd /home/energy_autoeval

# train
CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/datasets/ --dataset CIFAR-10 --num-classes 10 -j 4 \
--arch cifar10_vgg11_bn --epochs 200 --bs 256 --seed 1 --save-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_vgg11_bn/ \
--optimizer SGD -lr 0.1 --momentum 0.9 --nesterov -wd 5e-4 --scheduler CosineAnnealingLR --score EMD1 --T 1

# eval
CUDA_VISIBLE_DEVICES=0 python eval.py --data-dir /data/datasets/ --dataset CIFAR-10 --num-classes 10 -j 8 \
--arch cifar10_vgg11_bn --epoch 199 --bs 256 --save-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_vgg11_bn/ \
--ckpt-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_vgg11_bn/ --score EMD1 --T 1
