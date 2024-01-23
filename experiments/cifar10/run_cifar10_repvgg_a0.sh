#!/bin/bash
cd /home/energy_autoeval

CUDA_VISIBLE_DEVICES=0 python train.py --data-dir /data/datasets/ --dataset CIFAR-10 --num-classes 10 -j 4 \
--arch cifar10_repvgg_a0 --epochs 200 --bs 256 --seed 1 --save-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_repvgg_a0/ \
--optimizer SGD -lr 0.1 --momentum 0.9 --nesterov -wd 5e-4 --scheduler CosineAnnealingLR --score EMD1 --T 1

CUDA_VISIBLE_DEVICES=0 python eval.py --data-dir /data/datasets/ --dataset CIFAR-10 --num-classes 10 -j 8 \
--arch cifar10_repvgg_a0 --epoch 199 --bs 256 --save-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_repvgg_a0/ \
--ckpt-dir /data/checkpoints/energy_autoeval/cifar10/cifar10_repvgg_a0/ --score EMD1 --T 1
