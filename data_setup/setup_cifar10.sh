#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_cifar10c.sh <cifar_dir> <dataset>"
    exit 1
fi

if [ $2 == 'CIFAR10_1' ]; then
    ## Download CIFAR10_1
    echo "Downloading CIFAR10_1..."
    mkdir -p $1/CIFAR10_1

    wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v4_data.npy
    mv cifar10.1_v4_data.npy $1/CIFAR10_1/

    wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v4_labels.npy
    mv cifar10.1_v4_labels.npy $1/CIFAR10_1/

    wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_data.npy
    mv cifar10.1_v6_data.npy $1/CIFAR10_1/

    wget https://github.com/modestyachts/CIFAR-10.1/blob/master/datasets/cifar10.1_v6_labels.npy
    mv cifar10.1_v6_labels.npy $1/CIFAR10_1/

    echo "CIFAR10_1 downloaded"
fi

if [ $2 == 'CIFAR10_2' ]; then
    ## Download CIFAR10_2
    echo "Downloading CIFAR10_2..."
    mkdir -p $1/CIFAR10_2

    wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_train.npz
    mv cifar102_train.npz $1/CIFAR10_2/

    wget https://github.com/modestyachts/cifar-10.2/raw/master/cifar102_test.npz
    mv cifar102_test.npz $1/CIFAR10_2/

    echo "CIFAR10_2 downloaded"
fi

if [ $2 == 'CIFAR-10-C' ]; then

    ## Download CIFAR-10-C
    echo "Downloading CIFAR-10-C..."

    wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
    tar -xvf "CIFAR-10-C.tar?download=1" -C  $1/
    rm -rf "CIFAR-10-C.tar?download=1"

    echo "CIFAR-10-C downloaded"
fi

if [ $2 == 'CINIC10' ]; then
    ## Download CINIC10
    echo "Downloading CINIC10..."
    mkdir -p $1/CINIC10

    wget wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
    mv CINIC-10.tar.gz $1/CINIC10/

    echo "CINIC10 downloaded"
fi