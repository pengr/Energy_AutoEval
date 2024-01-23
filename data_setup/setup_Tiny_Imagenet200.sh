#!/bin/bash

### Note！！ ###
# This is just to create all the data sets of Tiny-ImageNet-200 Data Setup,
# and the pixel size has not been downsample to 64x64 (you need manually process it in datasets.py)
# More importantly, these 200 classes belong to the Tiny-ImageNet-200 are not equal to the ImageNet-200

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./setup_Tiny_Imagenet200.sh <data_dir>"
    exit 1
fi

IDS=('n01882714' 'n03424325' 'n04486054' 'n04285008' 'n09332890' 'n03599486' 'n04118538' 'n04254777' 'n03662601' 'n02843684' 'n02486410' 'n02056570' 'n03937543' 'n07583066' 'n07768694' 'n01945685' 'n02364673' 'n02206856' 'n04275548' 'n04501370' 'n03201208' 'n03089624' 'n02814533' 'n01944390' 'n02927161' 'n03838899' 'n04023962' 'n02814860' 'n02268443' 'n02125311' 'n02233338' 'n03014705' 'n02002724' 'n02793495' 'n04540053' 'n03670208' 'n04259630' 'n04596742' 'n02795169' 'n04328186' 'n02403003' 'n02129165' 'n02281406' 'n04251144' 'n01774384' 'n01917289' 'n03891332' 'n01774750' 'n03804744' 'n03992509' 'n04099969' 'n02236044' 'n03160309' 'n04366367' 'n07720875' 'n04560804' 'n01768244' 'n02999410' 'n02058221' 'n02231487' 'n07734744' 'n07873807' 'n04311004' 'n04398044' 'n04133789' 'n01855672' 'n04465501' 'n02788148' 'n02906734' 'n04376876' 'n03393912' 'n02917067' 'n04265275' 'n02124075' 'n02099712' 'n02977058' 'n02415577' 'n03902125' 'n03649909' 'n07875152' 'n02504458' 'n02892201' 'n07747607' 'n03100240' 'n02963159' 'n02802426' 'n01983481' 'n02395406' 'n07749582' 'n02791270' 'n03250847' 'n03026506' 'n03983396' 'n07715103' 'n02094433' 'n07695742' 'n06596364' 'n09193705' 'n02074367' 'n02099601' 'n02279972' 'n03042490' 'n04399382' 'n02808440' 'n02883205' 'n02085620' 'n02669723' 'n04179913' 'n03404251' 'n04532106' 'n04417672' 'n07614500' 'n03854065' 'n01784675' 'n02837789' 'n07615774' 'n09428293' 'n02132136' 'n02123045' 'n01644900' 'n03763968' 'n02481823' 'n04507155' 'n04146614' 'n02815834' 'n02909870' 'n04149813' 'n04597913' 'n03637318' 'n04067472' 'n02699494' 'n01770393' 'n03447447' 'n07871810' 'n04562935' 'n07753592' 'n03255030' 'n02730930' 'n04074963' 'n04487081' 'n03706229' 'n03355925' 'n02410509' 'n02823428' 'n02123394' 'n03976657' 'n04371430' 'n09256479' 'n01950731' 'n02321529' 'n02950826' 'n02948072' 'n01910747' 'n03400231' 'n01742172' 'n02509815' 'n01984695' 'n03544143' 'n04456115' 'n02113799' 'n03977966' 'n02106662' 'n03970156' 'n04532670' 'n02480495' 'n01641577' 'n04070727' 'n12267677' 'n09246464' 'n07711569' 'n01698640' 'n02190166' 'n03179701' 'n03837869' 'n02423022' 'n03126707' 'n07579787' 'n07920052' 'n03733131' 'n02226429' 'n03980874' 'n02437312' 'n02841315' 'n03814639' 'n03770439' 'n03796401' 'n02165456' 'n04008634' 'n03584254' 'n03444034' 'n01443537' 'n03388043' 'n02988304' 'n01629819' 'n03617480' 'n03085013' 'n04356056' 'n03930313' 'n02769748' 'n02666196')

# tiny-imagenet-200 train, val, 1000 class->200 class
echo "Download Tiny-Imagenet-200 by registering and following instrutions from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip

# tiny-imagenet-200v2-matched-frequency-format-val, 1000 class->200 class
mkdir -p $1/tiny-imagenet-200v2-matched-frequency-format-val
for id in ${IDS[@]}; do
    cp -rf $1/imagenetv2-matched-frequency-format-val/$id $1/tiny-imagenet-200v2-matched-frequency-format-val/$id
done

## tiny-imagenet-200v2-threshold0.7-format-val, 1000 class->200 class
mkdir -p $1/tiny-imagenet-200v2-threshold0.7-format-val
for id in ${IDS[@]}; do
    cp -rf $1/imagenetv2-threshold0.7-format-val/$id $1/tiny-imagenet-200v2-threshold0.7-format-val/$id
done

## tiny-imagenet-200v2-top-images-format-val, 1000 class->200 class
mkdir -p $1/tiny-imagenet-200v2-top-images-format-val
for id in ${IDS[@]}; do
    cp -rf $1/imagenetv2-top-images-format-val/$id $1/tiny-imagenet-200v2-top-images-format-val/$id
done

# Tiny-ImageNet-C, 1000 class->200 class
echo "Download Tiny-ImageNet-C by registering and following instrutions from https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1"
wget https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1

## tiny-imagenet-200-r, 200 class->62 class
mkdir -p $1/tiny-imagenet-200-r
for id in ${IDS[@]}; do
    cp -rf $1/imagenet-r/$id $1/tiny-imagenet-200-r/$id
done

# tiny-imagenet-200-sketch, 1000 class->200 class
mkdir -p $1/tiny-imagenet-200-sketch
for id in ${IDS[@]}; do
    cp -rf $1/imagenet-sketch/$id $1/tiny-imagenet-200-sketch/$id
done

# tiny-imagenet-200-vid-robust, 30 class->2 class
mkdir -p $1/tiny-imagenet-200-vid-robust
cp -rf $1/imagenet-vid-robust/metadata $1/tiny-imagenet-200-vid-robust/metadata
cp -rf $1/imagenet-vid-robust/misc $1/tiny-imagenet-200-vid-robust/misc
cp -rf $1/imagenet-vid-robust/readme.md $1/tiny-imagenet-200-vid-robust/readme.md
mkdir -p $1/tiny-imagenet-200-vid-robust/val
for id in ${IDS[@]}; do
    cp -rf $1/imagenet-vid-robust/val/$id $1/tiny-imagenet-200-vid-robust/val/$id
done

# tiny-imagenet-200-a, 200 class->74 class
mkdir -p $1/tiny-imagenet-200-a
for id in ${IDS[@]}; do
    cp -rf $1/imagenet-a/$id $1/tiny-imagenet-200-a/$id
done
