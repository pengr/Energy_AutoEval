#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_Imagenet.sh <imagenet_dir>"
    exit 1
fi

if [ -f "$1/imagenet_class_hierarchy/dataset_class_info.json" ]
then
    echo "OK"
else
   echo "Please download the BREEDs heirarcy first with the following command:"
   echo "./setup_BREEDs.sh ${1}"
fi

## Download Imagenet from here
echo "Download Imagenet by registering and following instrutions from http://image-net.org/download-images."

## Download Imagenetv2
echo "Downloading Imagenetv2..."
mkdir -p $1/imagenetv2

wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz
tar -xvf imagenetv2-matched-frequency.tar.gz  -C  $1/imagenetv2/  
rm -rf  imagenetv2-matched-frequency.tar.gz
#python data_setup/ImageNet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-matched-frequency-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json
python data_setup/ImageNet/ImageNet_v2_reorg.py --dir /data/datasets/imagenetv2-matched-frequency-format-val \
--info /home/codes/AutoEval/energy_autoeval/data_setup/imagenet_class_hierarchy/dataset_class_info.json

wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-threshold0.7.tar.gz
tar -xvf imagenetv2-threshold0.7.tar.gz -C  $1/imagenetv2/
rm -rf imagenetv2-threshold0.7.tar.gz
#python data_setup/ImageNet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-threshold0.7-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json
python data_setup/ImageNet/ImageNet_v2_reorg.py --dir /data/datasets/imagenetv2-threshold0.7-format-val \
--info /home/codes/AutoEval/energy_autoeval/data_setup/imagenet_class_hierarchy/dataset_class_info.json

wget https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-top-images.tar.gz
tar -xvf imagenetv2-top-images.tar.gz -C  $1/imagenetv2/
rm -rf imagenetv2-top-images.tar.gz
#python data_setup/ImageNet/ImageNet_v2_reorg.py --dir $1/imagenetv2/imagenetv2-top-images-format-val --info $1/imagenet_class_hierarchy/dataset_class_info.json
python data_setup/ImageNet/ImageNet_v2_reorg.py --dir /data/datasets/imagenetv2-top-images-format-val \
--info /home/codes/AutoEval/energy_autoeval/data_setup/imagenet_class_hierarchy/dataset_class_info.json
echo "Imagenetv2 downloaded"

## Download Imagenet C
echo "Downloading Imagenet C..."
mkdir -p $1/imagenet-c

wget https://zenodo.org/record/2235448/files/blur.tar?download=1 
tar -xvf "blur.tar?download=1" -C  $1/imagenet-c/
rm -rf "blur.tar?download=1"

wget https://zenodo.org/record/2235448/files/digital.tar?download=1
tar -xvf "digital.tar?download=1" -C  $1/imagenet-c/
rm -rf "digital.tar?download=1"

wget https://zenodo.org/record/2235448/files/extra.tar?download=1
tar -xvf "extra.tar?download=1" -C  $1/imagenet-c/
rm -rf "extra.tar?download=1"

wget https://zenodo.org/record/2235448/files/noise.tar?download=1
tar -xvf "noise.tar?download=1" -C  $1/imagenet-c/
rm -rf "noise.tar?download=1"

wget https://zenodo.org/record/2235448/files/weather.tar?download=1
tar -xvf "weather.tar?download=1" -C  $1/imagenet-c/
rm -rf "weather.tar?download=1"
echo "Imagenet C downloaded"

## Download Imagenet R
echo "Downloading Imagenet R..."
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
tar -xvf imagenet-r.tar -C  $1/
rm -rf imagenet-r.tar
echo "Imagenet R downloaded"

## Download Imagenet Sketch
echo "Downloading Imagenet Sketch..."
gdown https://drive.google.com/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA
unzip ImageNet-Sketch.zip -d $1/
mv $1/sketch/ $1/imagenet-sketch/
rm -rf ImageNet-Sketch.zip
echo "Imagenet Sketch downloaded"

## Download imagenet vid-robust
echo "Downloading Imagenet vid-robust..."
wget https://do-imagenet-classifiers-generalize-across-time.s3-us-west-2.amazonaws.com/imagenet_vid_ytbb_robust.tar.gz
tar -xvf imagenet_vid_ytbb_robust.tar.gz -C  $1/
mv $1/imagenet_vid_ytbb_robust/imagenet-vid-robust $1/
rm -rf imagenet_vid_ytbb_robust.tar.gz imagenet_vid_ytbb_robust

#python data_setup/ImageNet/ImageNet_v2_reorg.py --dir $1/imagenet_vid_robust/val/ --info $1/imagenet_class_hierarchy/dataset_class_info.json
python /home/codes/AutoEval/energy_autoeval/data_setup/ImageNet/ImageNet_vid_reorg.py --dir /data/datasets/imagenet-vid-robust/val/ \
--info1 /data/datasets/imagenet-vid-robust/metadata/labels.json \
--info2 /data/datasets/imagenet-vid-robust/misc/imagenet_vid_class_index.json
echo "Imagenet vid-robust downloaded"

## Download imagenet A
echo "Downloading Imagenet A..."
wget https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
tar -xvf imagenet-a.tar -C  $1/
rm -rf imagenet-a.tar
echo "Imagenet A downloaded"


    if eval:
        testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet/val/", transform=transform)
        testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-matched-frequency-format-val/", transform=transform)
        # testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-threshold0.7-format-val/", transform=transform)
        # testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2-top-images-format-val/", transform=transform)
        # testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-sketch/", transform=transform)
        # testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # # <fix>
        # testsetv4 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-r/", transform=transform)
        # testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv5 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-vid-robust/val", transform=transform)
        # testloaderv5 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        # testsetv6 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-a/", transform=transform)
        # testloaderv6 = torch.utils.data.DataLoader(testsetv5, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        #
        testsets.append(testsetv1)
        # testsets.append(testsetv2)
        # testsets.append(testsetv2_1)
        # testsets.append(testsetv2_2)
        # testsets.append(testsetv3)
        # testsets.append(testsetv4)
        # testsets.append(testsetv5)
        # testsets.append(testsetv6)
        #
        testloaders.append(testloaderv1)
        # testloaders.append(testloaderv2)
        # testloaders.append(testloaderv2_1)
        # testloaders.append(testloaderv2_2)
        # testloaders.append(testloaderv3)
        # testloaders.append(testloaderv4)
        # testloaders.append(testloaderv5)
        # testloaders.append(testloaderv6)

        testsetv7 = []
        testloaderv7 = []
        for data in imagenet_c:
            for severity in severities:
                testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity), transform=transform)
                testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

                testsetv7.append(testset)
                testloaderv7.append(testloader)
        testsets.append(testsetv7)
        testloaders.append(testloaderv7)

    return trainset, trainloader, testsets, testloaders
