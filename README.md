# Energy-based Automated Model Evaluation [[Paper]](https://openreview.net/pdf?id=CHGcP6lVWd)


## PyTorch Implementation

This repository contains:

- the PyTorch implementation of Energy_AutoEval
- the example on CIFAR-10 setup
- CIFAR-10, CIFAR-100, TinyImageNet-200, ImageNet-200, ImageNet download setups.
  Please see ```PROJECT_DIR/data_setup/``` or you can download it manually form the offical websites in Prerequisites ↓

Please follow the instruction below to install it and run the experiment demo.


### Prerequisites
* Linux (tested on Ubuntu 16.04LTS)
* NVIDIA GPU + CUDA CuDNN (tested on Tesla V100)
* [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/datasets/CIFAR10```)
* [CIFAR10.1 Dataset](https://github.com/modestyachts/CIFAR-10.1) (download and unzip to ```PROJECT_DIR/datasets/CIFAR10_1```)
* [CIFAR-10-C Dataset](https://zenodo.org/record/2535967#.Y-3ggHZBx3g) (download and unzip to ```PROJECT_DIR/datasets/CIFAR-10-C```)
* [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (download and unzip to ```PROJECT_DIR/datasets/CIFAR100```)
* [CIFAR-100-C Dataset](https://zenodo.org/record/3555552#.Y-3gwHZBx3g) (download and unzip to ```PROJECT_DIR/datasets/CIFAR-100-C```)
* [PASCAL Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) (download and unzip to ```PROJECT_DIR/datasets/PASCAL```)
* [Caltech256 Dataset](https://data.caltech.edu/records/nyy15-4j048) (download and unzip to ```PROJECT_DIR/datasets/Caltech256```)
* [ImageNet Dataset](https://image-net.org/challenges/LSVRC/2013/2013-downloads.php) (download and unzip to ```PROJECT_DIR/datasets/ImageNet```)
* [TinyImageNet Dataset](http://cs231n.stanford.edu/tiny-imagenet-200.zip) (download and unzip to ```PROJECT_DIR/datasets/tiny-imagenet-200```)
* [TinyImageNet-C Dataset](https://zenodo.org/record/2469796#.Y-3gynZBx3g) (download and unzip to ```PROJECT_DIR/datasets/Tiny-ImageNet-C```)
* SHVN,USPS,FashionMNIST,KMNIST,STL10 are the ready-made datasets in torchvision.datasets package 
* All -C Operation can refers to [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://github.com/hendrycks/robustness)
* You might need to change the file paths, and please be sure you change the corresponding paths in the codes as well  



## Getting started
0. Install dependencies 
    ```bash
    # Energy-based AutoEval
    conda env create --name energy_autoeval --file environment.yaml
    conda activate energy_autoeval
    ```
1. prepare datasets
    ```bash
    # download into "PROJECT_DIR/datasets/CIFAR10/"
    bash data_setup/setup_cifar10.sh
    ```

2. train classifier
    ```bash
    # Save as "PROJECT_DIR/checkpoints/CIFAR10/checkpoint.pth"
    python train.py
    ```

3. Eval on unseen test sets by regression model and Correlation study
    ```bash
    # The absolute error of linear regression is also shown
    python eval.py
    ``` 

        
## Citation
If you use the code in your research, please cite:
```bibtex
@inproceedings{peng2024energybased,
        title={Energy-based Automated Model Evaluation},
        author={Ru Peng, Heming Zou, Haobo Wang, Yawen Zeng, Zenan Huang, Junbo Zhao},
        booktitle={The Twelfth International Conference on Learning Representations},
        year={2024},
        url={https://openreview.net/forum?id=CHGcP6lVWd}
}
```

## License
MIT
