import pandas as pd
import torchvision.transforms as transforms
import csv
from utils import *
from transformers import BertTokenizerFast, RobertaTokenizerFast, DistilBertTokenizerFast
import imgaug.augmenters as iaa


def get_data(data_dir, data, bs, net, workers, train=True, eval=True, max_token_length=None):
    if data == "MNIST":
        return get_mnist(bs, workers, data_dir)

    if data == "CIFAR-10":
        return get_cifar10(bs, workers, data_dir)

    if data == "CIFAR-100":
        return get_cifar100(bs, workers, data_dir)

    if data == "ImageNet":
        return get_imagenet(bs, workers, net, data_dir, train, eval)

    if data == "ImageNet-200":
        return get_imagenet200(bs, workers, net, data_dir, train, eval)

    if data == "Tiny-ImageNet-200":
        return get_tiny_imagenet200(bs, workers, net, data_dir, train, eval)

    # if data == "living17" or data == "entity13" or data == "entity30" or data == "nonliving26":
    #     return get_imagenet_breeds(bs, workers, data_dir, data)

    if data == "MNLI":
        return get_mnli(bs, workers, net, data_dir, max_token_length, train, eval)


def get_mnist(batch_size, num_workers, data_dir):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    transform_gray = transforms.Compose([transforms.Grayscale(),
                                         transforms.Resize(28), # convert into the MNIST-type size
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root=f"{data_dir}/", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testsetv1 = torchvision.datasets.MNIST(root=f"{data_dir}/", train=False, download=True, transform=transform)
    testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testsetv2_1 = torchvision.datasets.SVHN(root=f"{data_dir}/", split='train', download=True, transform=transform_gray)
    testsetv2_2 = torchvision.datasets.SVHN(root=f"{data_dir}/", split='test', download=True, transform=transform_gray)
    testsetv2 = torch.utils.data.ConcatDataset([testsetv2_1, testsetv2_2])
    testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv3_1 = USPS(root=f"{data_dir}/", train=True, download=True, transform=transform)
    testsetv3_2 = USPS(root=f"{data_dir}/", train=False, download=True, transform=transform)
    testsetv3 = torch.utils.data.ConcatDataset([testsetv3_1, testsetv3_2])
    testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv4 = torchvision.datasets.QMNIST(root=f"{data_dir}/", train=False, download=True, transform=transform)
    testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    testsets.append(testsetv1)
    testsets.append(testsetv2)
    testsets.append(testsetv3)
    testsets.append(testsetv4)

    testloaders.append(testloaderv1)
    testloaders.append(testloaderv2)
    testloaders.append(testloaderv3)
    testloaders.append(testloaderv4)

    return trainset, trainloader, testsets, testloaders


def get_cifar10(batch_size, num_workers, data_dir):
    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
               "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
               "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4, 5]

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_cinic = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                                         transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])

    transform_stl10 = transforms.Compose([transforms.Resize((32, 32)), 
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    ## Transformed Tests
    list = [iaa.Cutout(nb_iterations=1),  ## Cutout
            iaa.Affine(shear=(-10, 10)),  ## Shear
            iaa.pillike.Equalize(),  ## Equalize
            iaa.ChangeColorTemperature((1100, 10000 // 2)),  ## ColorTemperature
            iaa.pillike.Posterize((1, 7)),  ## Posterize
            iaa.Pepper(0.01, 0.05),  ## Pepper
            iaa.pillike.FilterSmooth()]  ## FilterSmooth

    ## sort these transformed operations by the order: FD-A, FD-B, Rotation-A, Rotation-B
    transform_FD_A = iaa.Sequential([list[0], list[1]])
    transform_FD_B = iaa.Sequential([list[2], list[3]])
    transform_Rot_A = iaa.Sequential([list[0], list[4]])
    transform_Rot_B = iaa.Sequential([list[5], list[6]])
    transform_seqs = [transform_FD_A, transform_FD_B, transform_Rot_A, transform_Rot_B]

    trainset = torchvision.datasets.CIFAR10(root=f"{data_dir}/CIFAR10", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    # CIFAR-10 test
    testsetv1 = torchvision.datasets.CIFAR10(root=f"{data_dir}/CIFAR10", train=False, download=True, transform=transform_test)
    testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # CIFAR-10.1, it doesn't split into the train and test parts
    testsetv2 = CIFAR10v1(root=f"{data_dir}/CIFAR10_1/", download=True, transform=transform_test)
    testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # CIFAR-10.2
    testsetv3 = CIFAR10v2(root=f"{data_dir}/CIFAR10_2/", train=True, download=True, transform=transform_test)  # Train is true to get 10k points
    testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # CINIC-10
    testsetv4 = torchvision.datasets.ImageFolder(root=f"{data_dir}/CINIC10/test/", transform=transform_cinic)
    testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # STL-10
    # testsetv5_1 = torchvision.datasets.STL10(root=f"{data_dir}/", split='train', download=True, transform=transform_test)
    # testsetv5_2 = torchvision.datasets.STL10(root=f"{data_dir}/", split='test', download=True, transform=transform_test)
    # testsetv5 = torch.utils.data.ConcatDataset([testsetv5_1, testsetv5_2])
    # testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv5_1 = torchvision.datasets.STL10(root=f"{data_dir}/", split='train', download=True, transform=transform_stl10)
    testsetv5_2 = torchvision.datasets.STL10(root=f"{data_dir}/", split='test', download=True, transform=transform_stl10)
    testsetv5 = torch.utils.data.ConcatDataset([testsetv5_1, testsetv5_2])
    testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testsets.append(testsetv1)
    testsets.append(testsetv2)
    testsets.append(testsetv3)
    testsets.append(testsetv4)
    testsets.append(testsetv5)

    testloaders.append(testloaderv1)
    testloaders.append(testloaderv2)
    testloaders.append(testloaderv3)
    testloaders.append(testloaderv4)
    testloaders.append(testloaderv5)

    ## ImbalancedCIFAR10_C
    imb_factors = [0.1, 0.2, 0.4, 0.6, 0.8]
    for i in range(len(imb_factors)):
        testsetv6 = []
        testloaderv6 = []
        for data in cifar_c:
            # for severity in severities:
            testset = ImbalancedCIFAR10_C(root=f"{data_dir}/CIFAR-10-C/", cls_num=10, imb_factor=imb_factors[i],
                                          data_type=data, severity=2, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            testsets.append(testset)
            testloaders.append(testloader)
            # testsetv6.append(testset)
            # testloaderv6.append(testloader)
        # testsets.append(testsetv6)
        # testloaders.append(testloaderv6)

    # ## Transformed Unseen Test sets
    # testsetv7, testsetv8, testsetv9 = [], [], []
    # testloaderv7, testloaderv8, testloaderv9 = [], [], []
    # for seq in transform_seqs:
    #     ## TransformedCIFAR-10.1
    #     testset1 = TransformedCIFAR10v1(root=f"{data_dir}/CIFAR10_1/", download=True, transform=transform_test, transform1=seq)
    #     testloader1 = torch.utils.data.DataLoader(testset1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     testsetv7.append(testset1)
    #     testloaderv7.append(testloader1)
    #     # ## TransformedCIFAR-10.2
    #     testset2 = TransformedCIFAR10v2(root=f"{data_dir}/CIFAR10_2/", train=True, download=True, transform=transform_test, transform1=seq)
    #     testloader2 = torch.utils.data.DataLoader(testset2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     # testsetv8.append(testset2)
    #     # testloaderv8.append(testloader2)
    #     ## TransformedCINIC-10
    #     testset3 = TransformedCINIC10(root=f"{data_dir}/CINIC10/test", transform=transform_cinic, transform1=seq)
    #     testloader3 = torch.utils.data.DataLoader(testset3, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     # testsetv9.append(testset3)
    #     # testloaderv9.append(testloader3)
    #     # TransformedSTL-10, the original MAE is low, so we don't report the MAE of transformed version
    #     testset4_1 = TransformedSTL10(root=f"{data_dir}/", split='train', download=True, transform=transform_stl10, transform1=seq)
    #     testset4_2 = TransformedSTL10(root=f"{data_dir}/", split='test', download=True, transform=transform_stl10, transform1=seq)
    #     testset4 = torch.utils.data.ConcatDataset([testset4_1, testset4_2])
    #     testloader4 = torch.utils.data.DataLoader(testset4, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
    #     testsets.append(testset1)
    #     testsets.append(testset2)
    #     testsets.append(testset3)
    #     testsets.append(testset4)
    #     testloaders.append(testloader1)
    #     testloaders.append(testloader2)
    #     testloaders.append(testloader3)
    #     testloaders.append(testloader4)
    # # testsets.append(testsetv7)
    # # testsets.append(testsetv8)
    # # testsets.append(testsetv9)
    # # testsets.append(testsetv10)
    # # testloaders.append(testloaderv7)
    # # testloaders.append(testloaderv8)
    # # testloaders.append(testloaderv9)
    # # testloaders.append(testloaderv10)

    testsetv11 = []
    testloaderv11 = []
    for data in cifar_c:
        for severity in severities:
            testset = CIFAR10_C(root=f"{data_dir}/CIFAR-10-C/", data_type=data, severity=severity, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testsetv11.append(testset)
            testloaderv11.append(testloader)
    testsets.append(testsetv11)
    testloaders.append(testloaderv11)

    return trainset, trainloader, testsets, testloaders


def get_cifar100(batch_size, num_workers, data_dir):
    cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
               "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
               "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4, 5]

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = torchvision.datasets.CIFAR100(root=f"{data_dir}/CIFAR100", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    # CIFAR-100 test
    testsetv1 = torchvision.datasets.CIFAR100(root=f"{data_dir}/CIFAR100", train=False, download=True, transform=transform_test)
    testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets.append(testsetv1)
    testloaders.append(testloaderv1)

    # ## ImbalancedCIFAR100_C
    # imb_factors = [0.1, 0.2, 0.4, 0.6, 0.8]
    # for i in range(len(imb_factors)):
    #     testsetv2 = []
    #     testloaderv2 = []
    #     for data in cifar_c:
    #         # for severity in severities:
    #         testset = ImbalancedCIFAR10_C(root=f"{data_dir}/CIFAR-100-C/", cls_num=100, imb_factor=imb_factors[i],
    #                                       data_type=data, severity=2, transform=transform_test)
    #         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #         testsetv2.append(testset)
    #         testloaderv2.append(testloader)
    #     testsets.append(testsetv2)
    #     testloaders.append(testloaderv2)

    testsetv3 = []
    testloaderv3 = []
    for data in cifar_c:
        for severity in severities:
            testset = CIFAR10_C(root=f"{data_dir}/CIFAR-100-C/", data_type=data, severity=severity, transform=transform_test)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testsetv3.append(testset)
            testloaderv3.append(testloader)
    testsets.append(testsetv3)
    testloaders.append(testloaderv3)

    return trainset, trainloader, testsets, testloaders


def get_imagenet(batch_size, num_workers, net, data_dir, train=False, eval=True):
    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                  "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4, 5]

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if net.startswith("beit_base_patch16_224") or net.startswith("vit_base_patch16"):
        transform_test = transforms.Compose([transforms.Resize(int(224 / 0.9), interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif net.startswith("convnext_base"):
        transform_test = transforms.Compose([transforms.Resize(288, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif net.startswith("resnetv2_152x2_bit"):
        transform_test = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    elif net.startswith("swin_small_patch4"):
        transform_test = transforms.Compose([transforms.Resize(int(224 / 0.9), interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif net.startswith("densenet121") or net.startswith("densenet161") or net.startswith("densenet169") \
            or net.startswith("densenet201") or net.startswith("resnet50") or net.startswith("resnet101") \
            or net.startswith("resnet152") or net.startswith("vgg16") or net.startswith("vgg19"):
        transform_test = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform_test = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = None
    trainloader = None

    if train:
        trainset = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet/train/", transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    if eval:
        testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet/val/", transform=transform_test)
        testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-matched-frequency-format-val/", transform=transform_test)
        # testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-threshold0.7-format-val/", transform=transform_test)
        # testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-top-images-format-val/", transform=transform_test)
        # testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-sketch/", transform=transform_test)
        # testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv4 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-r/", transform=transform_test)
        # testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv5 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-vid-robust/val", transform=transform_test)
        # testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv6 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-a/", transform=transform_test)
        # testloaderv6 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsets.append(testsetv1)
        # testsets.append(testsetv2)
        # testsets.append(testsetv2_1)
        # testsets.append(testsetv2_2)
        # testsets.append(testsetv3)
        # testsets.append(testsetv4)
        # testsets.append(testsetv5)
        # testsets.append(testsetv6)

        testloaders.append(testloaderv1)
        # testloaders.append(testloaderv2)
        # testloaders.append(testloaderv2_1)
        # testloaders.append(testloaderv2_2)
        # testloaders.append(testloaderv3)
        # testloaders.append(testloaderv4)
        # testloaders.append(testloaderv5)
        # testloaders.append(testloaderv6)

        # imb_factors = [0.1, 0.2, 0.4, 0.6, 0.8]
        # for i in range(len(imb_factors)):
        #     testsetv7 = []
        #     testloaderv7 = []
        #     for data in imagenet_c:
        #         # for severity in severities:
        #         testset = ImbalancedImageNet_C(root=f"{data_dir}/imagenet-c/" + data + "/" + str(2), cls_num=1000,
        #                                        imb_factor=imb_factors[i], transform=transform_test)
        #         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        #         testsetv7.append(testset)
        #         testloaderv7.append(testloader)
        #     testsets.append(testsetv7)
        #     testloaders.append(testloaderv7)

        testsetv8 = []
        testloaderv8 = []
        for data in imagenet_c:
            for severity in severities:
                testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity), transform=transform_test)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                testsetv8.append(testset)
                testloaderv8.append(testloader)
        testsets.append(testsetv8)
        testloaders.append(testloaderv8)

    return trainset, trainloader, testsets, testloaders


def get_imagenet200(batch_size, num_workers, net, data_dir, train=False, eval=True):
    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                  "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4, 5]

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    trainset = None
    trainloader = None

    if train:
        trainset = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200/train/", transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    if eval:
        testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200/val/", transform=transform_test)
        testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200v2-matched-frequency-format-val/", transform=transform_test)
        testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200v2-threshold0.7-format-val/", transform=transform_test)
        testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200v2-top-images-format-val", transform=transform_test)
        testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200-sketch/", transform=transform_test)
        testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv4 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200-r/", transform=transform_test)
        testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv5 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200-vid-robust/val/", transform=transform_test)
        testloaderv5 = torch.utils.data.testsetv4(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv6 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet200-a/", transform=transform_test)
        testloaderv6 = torch.utils.data.testsetv4(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsets.append(testsetv1)
        testsets.append(testsetv2)
        testsets.append(testsetv2_1)
        testsets.append(testsetv2_2)
        testsets.append(testsetv3)
        testsets.append(testsetv4)
        testsets.append(testsetv5)
        testsets.append(testsetv6)

        testloaders.append(testloaderv1)
        testloaders.append(testloaderv2)
        testloaders.append(testloaderv2_1)
        testloaders.append(testloaderv2_2)
        testloaders.append(testloaderv3)
        testloaders.append(testloaderv4)
        testloaders.append(testloaderv5)
        testloaders.append(testloaderv6)

        imb_factors = [0.1, 0.2, 0.4, 0.6, 0.8]
        for i in range(len(imb_factors)):
            testsetv7 = []
            testloaderv7 = []
            for data in imagenet_c:
                # for severity in severities:
                testset = ImbalancedImageNet_C(root=f"{data_dir}/imagenet200-c/" + data + "/" + str(2), cls_num=200,
                                               imb_factor=imb_factors[i], transform=transform_test)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                testsetv7.append(testset)
                testloaderv7.append(testloader)
            testsets.append(testsetv7)
            testloaders.append(testloaderv7)

        testsetv8 = []
        testloaderv8 = []
        for data in imagenet_c:
            for severity in severities:
                testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet200-c/" + data + "/" + str(severity), transform=transform_test)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                testsetv8.append(testset)
                testloaderv8.append(testloader)
        testsets.append(testsetv8)
        testloaders.append(testloaderv8)

    return trainset, trainloader, testsets, testloaders


def get_tiny_imagenet200(batch_size, num_workers, net, data_dir, train=False, eval=True):
    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                  "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                  "jpeg_compression"]
    severities = [1, 2, 3, 4, 5]

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_test = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = None
    trainloader = None

    if train:
        trainset = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200/train/", transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    if eval:
        testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200/val/", transform=transform_test)
        testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200v2-matched-frequency-format-val/", transform=transform_test)
        testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200v2-threshold0.7-format-val/", transform=transform_test)
        testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200v2-top-images-format-val", transform=transform_test)
        testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200-sketch/", transform=transform_test)
        testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv4 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200-r/", transform=transform_test)
        testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv5 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200-vid-robust/val/", transform=transform_test)
        testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv6 = torchvision.datasets.ImageFolder(f"{data_dir}/tiny-imagenet-200-a/", transform=transform_test)
        testloaderv6 = torch.utils.data.DataLoader(testsetv6, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsets.append(testsetv1)
        testsets.append(testsetv2)
        testsets.append(testsetv2_1)
        testsets.append(testsetv2_2)
        testsets.append(testsetv3)
        testsets.append(testsetv4)
        testsets.append(testsetv5)
        testsets.append(testsetv6)

        testloaders.append(testloaderv1)
        testloaders.append(testloaderv2)
        testloaders.append(testloaderv2_1)
        testloaders.append(testloaderv2_2)
        testloaders.append(testloaderv3)
        testloaders.append(testloaderv4)
        testloaders.append(testloaderv5)
        testloaders.append(testloaderv6)

        # imb_factors = [0.1, 0.2, 0.4, 0.6, 0.8]
        # for i in range(len(imb_factors)):
        #     testsetv7 = []
        #     testloaderv7 = []
        #     for data in imagenet_c:
        #         # for severity in severities:
        #         testset = ImbalancedImageNet_C(root=f"{data_dir}/tiny-imagenet-200-c/" + data + "/" + str(2), cls_num=200,
        #                                        imb_factor=imb_factors[i], transform=transform_test)
        #         testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        #         testsetv7.append(testset)
        #         testloaderv7.append(testloader)
        #     testsets.append(testsetv7)
        #     testloaders.append(testloaderv7)

        testsetv8 = []
        testloaderv8 = []
        for data in imagenet_c:
            for severity in severities:
                testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/tiny-imagenet-200-c/" + data + "/" + str(severity), transform=transform_test)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                testsetv8.append(testset)
                testloaderv8.append(testloader)
        testsets.append(testsetv8)
        testloaders.append(testloaderv8)

    return trainset, trainloader, testsets, testloaders


def get_imagenet_breeds(batch_size, num_workers, data_dir, name=None):
    from robustness.tools.helpers import get_label_mapping
    from robustness.tools import folder
    from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26

    if name == "living17":
        ret = make_living17(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif name == "entity13":
        ret = make_entity13(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif name == "entity30":
        ret = make_entity30(f"{data_dir}/imagenet_class_hierarchy/", split="good")
    elif name == "nonliving26":
        ret = make_nonliving26(f"{data_dir}/imagenet_class_hierarchy/", split="good")

    keep_ids = np.array(ret[1]).reshape((-1))

    # merge_label_mapping = get_label_mapping('custom_imagenet', np.concatenate((ret[1][0], ret[1][1]), axis=1))
    source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0])
    target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])])

    trainset = None
    trainloader = None

    trainset = folder.ImageFolder(root=f"{data_dir}/imagenetv1/train/", transform=transform,
                                  label_mapping=source_label_mapping)
    testset = folder.ImageFolder(root=f"{data_dir}/imagenetv1/train/", transform=transform,
                                 label_mapping=target_label_mapping)

    imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur", \
                  "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate"]
    severities = [1, 2, 3, 4, 5]

    idx = np.arange(len(trainset))
    np.random.seed(42)
    np.random.shuffle(idx)

    train_idx = idx[:len(idx) - 10000]
    val_idx = idx[len(idx) - 10000:]

    train_subset = torch.utils.data.Subset(trainset, train_idx)
    test_subset = torch.utils.data.Subset(trainset, val_idx)

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []

    testloaderv0_1 = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloaderv0_2 = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv1 = folder.ImageFolder(f"{data_dir}/imagenetv1/val/", transform=transform, label_mapping=source_label_mapping)
    testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv2 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/",
                                   transform=transform, label_mapping=source_label_mapping)
    testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv2_1 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform=transform,
                                     label_mapping=source_label_mapping)
    testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv2_2 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform=transform,
                                     label_mapping=source_label_mapping)
    testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets.append(test_subset)
    testsets.append(testset)
    testsets.append(testsetv1)
    testsets.append(testsetv2)
    testsets.append(testsetv2_1)
    testsets.append(testsetv2_2)

    testloaders.append(testloaderv0_1)
    testloaders.append(testloaderv0_2)
    testloaders.append(testloaderv1)
    testloaders.append(testloaderv2)
    testloaders.append(testloaderv2_1)
    testloaders.append(testloaderv2_2)

    testsetv3 = []
    testloaderv3 = []
    for data in imagenet_c:
        for severity in severities:
            testset = folder.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity), \
                                         transform=transform, label_mapping=source_label_mapping)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, \
                                                     num_workers=num_workers)
            testsetv3.append(testset)
            testloaderv3.append(testloader)

    testsetv1_t = folder.ImageFolder(f"{data_dir}/imagenetv1/val/", transform=transform,
                                     label_mapping=target_label_mapping)
    testloaderv1_t = torch.utils.data.DataLoader(testsetv1_t, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv2_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/",
                                     transform=transform, label_mapping=target_label_mapping)
    testloaderv2_t = torch.utils.data.DataLoader(testsetv2_t, batch_size=batch_size, shuffle=True,num_workers=num_workers)

    testsetv2_1_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/",
                                       transform=transform, label_mapping=target_label_mapping)
    testloaderv2_1_t = torch.utils.data.DataLoader(testsetv2_1_t, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsetv2_2_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform=transform,
                                       label_mapping=target_label_mapping)
    testloaderv2_2_t = torch.utils.data.DataLoader(testsetv2_2_t, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets.append(testsetv1_t)
    testsets.append(testsetv2_t)
    testsets.append(testsetv2_1_t)
    testsets.append(testsetv2_2_t)

    testloaders.append(testloaderv1_t)
    testloaders.append(testloaderv2_t)
    testloaders.append(testloaderv2_1_t)
    testloaders.append(testloaderv2_2_t)

    for data in imagenet_c:
        for severity in severities:
            testset_t = folder.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity), \
                                           transform=transform, label_mapping=target_label_mapping)
            testloader_t = torch.utils.data.DataLoader(testset_t, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            testsetv3.append(testset_t)
            testloaderv3.append(testloader_t)
    testloaders.append(testsetv3)
    testloaders.append(testloaderv3)

    return trainset, trainloader, testsets, testloaders


def getBertTokenizer(model, max_token_length):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model, truncation=True, max_length=max_token_length)
    elif model == 'roberta-base':
        tokenizer = RobertaTokenizerFast.from_pretrained(model, truncation=True, max_length=max_token_length)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model, truncation=True, max_length=max_token_length)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer


def initialize_bert_transform(net, max_token_length=512):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None

    tokenizer = getBertTokenizer(net, max_token_length)

    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        if net == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif net == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1
        return x

    return transform


def get_mnli(batch_size, num_workers, net, data_dir, max_token_length, train=False, eval=True):
    while True:
        try:
            tokenizer = getBertTokenizer(net, max_token_length)
            break
        except Exception:
            continue

    trainset = None
    trainloader = None
    if train:
        ## MNLI: 392702 lines
        train_df = pd.read_csv(f"{data_dir}/MNLI/train.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        trainset = BertDataset(train_df, tokenizer, max_token_length)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testsets = []
    testloaders = []
    if eval:  ## Determine which data set is used, you can see Table 7 of Zhang et al.,2020
        ## MNLI-M: 9815 lines
        testsetv1_df = pd.read_csv(f"{data_dir}/MNLI/dev_matched.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv1 = BertDataset(testsetv1_df, tokenizer, max_token_length)
        testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## MNLI-MM: 9832 lines
        testsetv2_df = pd.read_csv(f"{data_dir}/MNLI/dev_mismatched.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv2 = BertDataset(testsetv2_df, tokenizer, max_token_length)
        testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## SNLI: {snli_1.0_train.jsonl: 550152 lines, snli_1.0_dev.jsonl: 10000 lines. snli_1.0_test.jsonl: 10000 lines}
        testsetv3_df = pd.read_json(f"{data_dir}/SNLI/snli_1.0_dev.jsonl", lines=True)
        testsetv3 = BertDataset(testsetv3_df, tokenizer, max_token_length)
        testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Breaking_NLI: 8193 lines
        testsetv4_df = pd.read_json(f"{data_dir}/Breaking_NLI/data/dataset.jsonl", lines=True)
        testsetv4 = BertDataset(testsetv4_df, tokenizer, max_token_length)
        testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## HANS: {heuristics_train_set: 30000 lines, heuristics_evaluation_set: 30000 lines}
        testsetv5_df = pd.read_json(f"{data_dir}/HANS/heuristics_evaluation_set.jsonl", lines=True)
        testsetv5 = HANSBertDataset(testsetv5_df, tokenizer, max_token_length)
        testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## SNLI-hard: 3261 lines
        testsetv6_df = pd.read_json(f"{data_dir}/SNLI/snli_1.0_test_hard.jsonl", lines=True)
        testsetv6 = BertDataset(testsetv6_df, tokenizer, max_token_length)
        testloaderv6 = torch.utils.data.DataLoader(testsetv6, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-L: length_mismatch_matched: 9815 lines, length_mismatch_mismatched: 9832 lines
        testsetv7_df = pd.read_json(f"{data_dir}/Stress-Tests/Length_Mismatch/multinli_0.9_length_mismatch_matched.jsonl", lines=True)
        testsetv7 = BertDataset(testsetv7_df, tokenizer, max_token_length)
        testloaderv7 = torch.utils.data.DataLoader(testsetv7, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-S: gram_contentword_swap_perturbed_matched 8243 lines, gram_contentword_swap_perturbed_mismatched: 6824 lines
        testsetv8_df = pd.read_json(f"{data_dir}/Stress-Tests/Spelling_Error/multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl", lines=True)
        testsetv8 = BertDataset(testsetv8_df, tokenizer, max_token_length)
        testloaderv8 = torch.utils.data.DataLoader(testsetv8, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-NE: negation_matched: 9815 lines, negation_mismatched: 9832 lines
        # testsetv8_df = pd.read_json(f"{data_dir}/Stress-Tests/Negation/multinli_0.9_negation_matched.jsonl", lines=True)
        # testsetv8 = BertDataset(testsetv8_df, tokenizer, max_token_length)
        # testloaderv8 = torch.utils.data.DataLoader(testsetv8, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv8_df1 = pd.read_json(f"{data_dir}/Stress-Tests/Negation/multinli_0.9_negation_mismatched.jsonl", lines=True)
        # testsetv8_1 = BertDataset(testsetv8_df1, tokenizer, max_token_length)
        # testloaderv8_1 = torch.utils.data.DataLoader(testsetv8_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-O: taut2_matched: 9815 lines, taut2_mismatched: 9832 lines
        testsetv9_df = pd.read_json(f"{data_dir}/Stress-Tests/Word_Overlap/multinli_0.9_taut2_matched.jsonl", lines=True)
        testsetv9 = BertDataset(testsetv9_df, tokenizer, max_token_length)
        testloaderv9 = torch.utils.data.DataLoader(testsetv9, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv9_df1 = pd.read_json(f"{data_dir}/Stress-Tests/Word_Overlap/multinli_0.9_taut2_mismatched.jsonl", lines=True)
        # testsetv9_1 = BertDataset(testsetv9_df1, tokenizer, max_token_length)
        # testloaderv9_1 = torch.utils.data.DataLoader(testsetv9_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-NU: quant_hard: 7596 lines
        # testsetv10_df = pd.read_json(f"{data_dir}/Stress-Tests/Numerical_Reasoning/multinli_0.9_quant_hard.jsonl", lines=True)
        # testsetv10 = BertDataset(testsetv10_df, tokenizer, max_token_length)
        # testloaderv10 = torch.utils.data.DataLoader(testsetv10, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## SICK: {train+dev+test, 9841 lines}
        # testsetv11_df = pd.read_csv(f"{data_dir}/SICK/SICK.txt", sep='\t', quoting=csv.QUOTE_NONE)
        # testsetv11_df.rename(columns={'sentence_A': 'sentence1', 'sentence_B': 'sentence2', 'entailment_label': 'gold_label'}, inplace=True)
        # testsetv11 = SICKBertDataset(testsetv11_df, tokenizer, max_token_length)
        # testloaderv11 = torch.utils.data.DataLoader(testsetv11, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv11_df1 = pd.read_csv(f"{data_dir}/SICK/SICK.txt", sep='\t', quoting=csv.QUOTE_NONE)
        # testsetv11_df1 = testsetv11_df1[testsetv11_df1.iloc[:, -1] == 'TRAIN']
        # testsetv11_df1.rename(columns={'sentence_A': 'sentence1', 'sentence_B': 'sentence2', 'entailment_label': 'gold_label'}, inplace=True)
        # testsetv11_1 = SICKBertDataset(testsetv11_df1, tokenizer, max_token_length)
        # testloaderv11_1 = torch.utils.data.DataLoader(testsetv11_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsetv11_df2 = pd.read_csv(f"{data_dir}/SICK/SICK.txt", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv11_df2 = testsetv11_df2[testsetv11_df2.iloc[:, -1] == 'TRIAL']
        testsetv11_df2.rename(columns={'sentence_A': 'sentence1', 'sentence_B': 'sentence2', 'entailment_label': 'gold_label'}, inplace=True)
        testsetv11_2 = SICKBertDataset(testsetv11_df2, tokenizer, max_token_length)
        testloaderv11_2 = torch.utils.data.DataLoader(testsetv11_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        # testsetv11_df3 = pd.read_csv(f"{data_dir}/SICK/SICK.txt", sep='\t', quoting=csv.QUOTE_NONE)
        # testsetv11_df3 = testsetv11_df3[testsetv11_df3.iloc[:, -1] == 'TEST']
        # testsetv11_df3.rename(columns={'sentence_A': 'sentence1', 'sentence_B': 'sentence2', 'entailment_label': 'gold_label'}, inplace=True)
        # testsetv11_3 = SICKBertDataset(testsetv11_df3, tokenizer, max_token_length)
        # testloaderv11_3 = torch.utils.data.DataLoader(testsetv11_3, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## EQUATE-SYN: {AWPNLI: 722 lines, StressTest: 7596 lines, All: 8318 lines}
        # testsetv12_df1 = pd.read_json(f"{data_dir}/EQUATE/AWPNLI.jsonl", lines=True)
        # testsetv12_df2 = pd.read_json(f"{data_dir}/EQUATE/StressTest.jsonl", lines=True)
        # testsetv12_1 = BertDataset(testsetv12_df1, tokenizer, max_token_length)
        # testsetv12_2 = BertDataset(testsetv12_df2, tokenizer, max_token_length)
        # testsetv12 = torch.utils.data.ConcatDataset([testsetv12_1, testsetv12_2])
        # testloaderv12 = torch.utils.data.DataLoader(testsetv12, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## Stress-Tests-A: antonym_matched: 1561 lines, antonym_mismatched: 1734 lines
        testsetv10_df = pd.read_json(f"{data_dir}/Stress-Tests/Antonym/multinli_0.9_antonym_matched.jsonl", lines=True)
        testsetv10 = BertDataset(testsetv10_df, tokenizer, max_token_length)
        testloaderv10 = torch.utils.data.DataLoader(testsetv10, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## EQUATE-NAT: {NewsNLI: 968 lines, RedditNLI: 250 lines, RTE_Quant: 166 lines, All: 1384 lines}
        testsetv12_df1 = pd.read_json(f"{data_dir}/EQUATE/NewsNLI.jsonl", lines=True)
        testsetv12_df2 = pd.read_json(f"{data_dir}/EQUATE/RedditNLI.jsonl", lines=True)
        testsetv12_df3 = pd.read_json(f"{data_dir}/EQUATE/RTE_Quant.jsonl", lines=True)
        testsetv12_1 = BertDataset(testsetv12_df1, tokenizer, max_token_length)
        testsetv12_2 = BertDataset(testsetv12_df2, tokenizer, max_token_length)
        testsetv12_3 = BertDataset(testsetv12_df3, tokenizer, max_token_length)
        testsetv12 = torch.utils.data.ConcatDataset([testsetv12_1, testsetv12_2, testsetv12_3])
        testloaderv12 = torch.utils.data.DataLoader(testsetv12, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## QNLI: {train.tsv: 104743 lines, dev.tsv: 5453 lines, test.tsv(without label): 5463 lines}
        testsetv16_df = pd.read_csv(f"{data_dir}/QNLI/dev.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv16_df.rename(columns={'question': 'sentence1', 'sentence': 'sentence2', 'label': 'gold_label'}, inplace=True)
        testsetv16 = QNLIBertDataset(testsetv16_df, tokenizer, max_token_length)
        testloaderv16 = torch.utils.data.DataLoader(testsetv16, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## RTE: {train.tsv: 2490 lines, dev.tsv: 277 lines, test.tsv(without label): 3000 lines}
        testsetv17_df = pd.read_csv(f"{data_dir}/RTE/dev.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv17_df.rename(columns={'label': 'gold_label'}, inplace=True)
        testsetv17 = QNLIBertDataset(testsetv17_df, tokenizer, max_token_length)
        testloaderv17 = torch.utils.data.DataLoader(testsetv17, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## WNLI: {train.tsv: 635 lines, dev.tsv: 71 lines, test.tsv(without label): 146 lines}
        testsetv18_df = pd.read_csv(f"{data_dir}/WNLI/dev.tsv", sep='\t', quoting=csv.QUOTE_NONE)
        testsetv18_df.rename(columns={'label': 'gold_label'}, inplace=True)
        testsetv18 = WNLIBertDataset(testsetv18_df, tokenizer, max_token_length)
        testloaderv18 = torch.utils.data.DataLoader(testsetv18, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## ANLI-R1: {train.tsv: 16946 lines, dev.tsv: 1000 lines, test.tsv: 1000 lines}
        testsetv19_df = pd.read_json(f"{data_dir}/ANLI/R1/dev.jsonl", lines=True)
        testsetv19_df.rename(columns={'context': 'sentence1', 'hypothesis': 'sentence2', 'label': 'gold_label'}, inplace=True)
        testsetv19 = ANLIBertDataset(testsetv19_df, tokenizer, max_token_length)
        testloaderv19 = torch.utils.data.DataLoader(testsetv19, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## ANLI-R2: {train.tsv: 45460 lines, dev.tsv: 1000 lines, test.tsv: 1000 lines}
        testsetv20_df = pd.read_json(f"{data_dir}/ANLI/R2/dev.jsonl", lines=True)
        testsetv20_df.rename(columns={'context': 'sentence1', 'hypothesis': 'sentence2', 'label': 'gold_label'}, inplace=True)
        testsetv20 = ANLIBertDataset(testsetv20_df, tokenizer, max_token_length)
        testloaderv20 = torch.utils.data.DataLoader(testsetv20, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## ANLI-R3: {train.tsv: 100459 lines, dev.tsv: 1200 lines, test.tsv: 1200 lines}
        testsetv21_df = pd.read_json(f"{data_dir}/ANLI/R3/dev.jsonl", lines=True)
        testsetv21_df.rename(columns={'context': 'sentence1', 'hypothesis': 'sentence2', 'label': 'gold_label'}, inplace=True)
        testsetv21 = ANLIBertDataset(testsetv21_df, tokenizer, max_token_length)
        testloaderv21 = torch.utils.data.DataLoader(testsetv21, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        ## SciTail: train.tsv: 23596 lines, dev.tsv: 1304 lines, test.tsv: 2126 lines
        testsetv22_df = pd.read_csv(f"{data_dir}/SciTail/tsv_format/scitail_1.0_dev.tsv", sep='\t', quoting=csv.QUOTE_NONE, header=None)
        testsetv22_df.rename(columns={0: 'sentence1', 1: 'sentence2', 2: 'gold_label'}, inplace=True)
        testsetv22 = SciTailBertDataset(testsetv22_df, tokenizer, max_token_length)
        testloaderv22 = torch.utils.data.DataLoader(testsetv22, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testsets.append(testsetv16)
        testsets.append(testsetv17)
        testsets.append(testsetv18)
        testsets.append(testsetv19)
        testsets.append(testsetv20)
        testsets.append(testsetv21)
        testsets.append(testsetv22)
        testset_oods = [testsetv1, testsetv2, testsetv3, testsetv4, testsetv5, testsetv6, testsetv7, testsetv8, testsetv9,
                        testsetv10, testsetv11_2, testsetv12]
        testsets.append(testset_oods)

        testloaders.append(testloaderv16)
        testloaders.append(testloaderv17)
        testloaders.append(testloaderv18)
        testloaders.append(testloaderv19)
        testloaders.append(testloaderv20)
        testloaders.append(testloaderv21)
        testloaders.append(testloaderv22)
        testloader_oods = [testloaderv1, testloaderv2, testloaderv3, testloaderv4, testloaderv5, testloaderv6, testloaderv7, testloaderv8, testloaderv9,
                           testloaderv10, testloaderv11_2, testloaderv12]
        testloaders.append(testloader_oods)

    return trainset, trainloader, testsets, testloaders