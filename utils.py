from PIL import Image
import gzip
import os
import pickle
import urllib
import numpy as np
import torchvision
import torch
import shutil
import yaml
import torch.utils.data as data
from torchvision import transforms
import imgaug
import imgaug.augmenters
import random
from itertools import chain

class RandomSplit():
    def __init__(self, dataset, indices):
        self.dataset = dataset

        self.size = len(indices)
        self.indices = indices

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        out = self.dataset[self.indices[index]]

        return out


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CIFAR10v1(torchvision.datasets.CIFAR10):

    def __init__(self, root, transform=None, target_transform=None, version='v6',
                 download=False):
        self.data = np.load('%s/cifar10.1_%s_data.npy' % (root, version))
        self.targets = np.load('%s/cifar10.1_%s_labels.npy' % (root, version)).astype('long')
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10v2(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            data = np.load(root + "/" + 'cifar102_train.npz', allow_pickle=True)
        else:
            data = np.load(root + "/" + 'cifar102_test.npz', allow_pickle=True)

        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_C(torchvision.datasets.CIFAR10):

    def __init__(self, root, data_type=None, severity=1, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        data = np.load(root + "/" + data_type + '.npy')
        labels = np.load(root + "/" + 'labels.npy')

        self.data = data[(severity - 1) * 10000: (severity) * 10000]
        self.targets = labels[(severity - 1) * 10000: (severity) * 10000].astype(np.int_)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BinaryCIFARv2(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            data = np.load(root + "/" + 'cifar102_train.npy', allow_pickle=True).item()
        else:
            data = np.load(root + "/" + 'cifar102_test.npy', allow_pickle=True).item()

        self.data = data["images"]
        self.targets = (data["labels"] >= 5).astype(np.int_)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BinaryCIFAR(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform,
                         download)

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.targets = (self.targets >= 5).astype(np.int_)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFARMixture(torchvision.datasets.CIFAR10):

    def __init__(self, clean_data, random_data, num_classes=2, transform=None):
        self.target_transform = None
        self.transform = transform

        random_size = len(random_data)
        random_labels = np.random.randint(low=0, high=num_classes, size=random_size, dtype=np.int_)

        # np.random.shuffle(random_labels)

        self.data = np.concatenate((clean_data.data, random_data.data), axis=0)
        self.targets = np.concatenate((clean_data.targets, random_labels), axis=0)
        self.true_targets = np.concatenate((clean_data.targets, random_data.targets), axis=0)

        self.flipped = np.zeros_like(self.targets)
        self.true_mask = np.zeros_like(self.targets)

        self.flipped[-random_size:] = 1.0
        self.true_mask[:-random_size] = 1.0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        flip = self.flipped[index]
        true_mask = self.true_mask[index]
        true_targets = self.true_targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        # target = self.target_transform(target)

        return img, target, flip, true_mask, true_targets


class USPS(torch.utils.data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        # img =
        # print(img.shape)
        img = Image.fromarray(img.squeeze().astype(np.int8), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]
        return images, labels


## Long-tailed Imbalanced Test sets
class ImbalancedCIFAR10_C(CIFAR10_C):
    def __init__(self, root, cls_num=10, imb_type='exp', imb_factor=0.01, data_type=None, severity=1, transform=None,
                 target_transform=None, download=False):
        super(ImbalancedCIFAR10_C, self).__init__(root, data_type, severity, transform, target_transform, download)
        # np.random.seed(rand_number)
        self.cls_num = cls_num
        img_num_list = self.get_img_num_per_cls(imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, imb_type, imb_factor):
        img_max = len(self.data) / self.cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class ImbalancedImageNet_C(torchvision.datasets.ImageFolder):
    def __init__(self, root, cls_num=1000, imb_type='exp', imb_factor=0.01, transform=None, target_transform=None):
        super(ImbalancedImageNet_C, self).__init__(root, transform, target_transform)
        # np.random.seed(rand_number)
        self.cls_num = cls_num
        img_num_list = self.get_img_num_per_cls(imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, imb_type, imb_factor):
        img_max = len(self.data) / self.cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = img_max * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(self.cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * self.cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data, new_targets = [], []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

## Transformed Test sets
class TransformedCIFAR10v1(torchvision.datasets.CIFAR10):

    def __init__(self, root, transform=None, transform1=None, target_transform=None, version='v6',
                 download=False):
        self.data = np.load('%s/cifar10.1_%s_data.npy' % (root, version))
        self.targets = np.load('%s/cifar10.1_%s_labels.npy' % (root, version)).astype('long')
        self.transform = transform
        self.transform1 = transform1
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform1 is not None: # input: 32 * 32 * 3
            img = self.transform1(image=img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformedCIFAR10v2(torchvision.datasets.CIFAR10):

    def __init__(self, root, train=True, transform=None, transform1=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.transform1 = transform1
        self.target_transform = target_transform

        if train:
            data = np.load(root + "/" + 'cifar102_train.npz', allow_pickle=True)
        else:
            data = np.load(root + "/" + 'cifar102_test.npz', allow_pickle=True)

        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self):
        return len(self.targets)

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        # if self.transform is not None:
        #     img = self.transform(img)

        # img = np.array(img).transpose(1, 2, 0).astype(np.uint8)
        # if self.transform1 is not None:
        #     img = self.transform1(image=img)
        #     img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # return img, target
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform1 is not None:
            img = self.transform1(image=img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class TransformedCINIC10(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, transform1=None, target_transform=None):
        super(TransformedCINIC10, self).__init__(root, transform, target_transform)
        self.transform1 = transform1

    def loader(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = np.array(self.loader(path)) # 32*32*3
        # sample = Image.fromarray(sample)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # sample = np.array(sample).transpose(1, 2, 0).astype(np.uint8)
        # if self.transform1 is not None:
        #     sample = self.transform1(image=sample)
        #     sample = torch.from_numpy(sample.transpose(2, 0, 1)).to(torch.float)
        if self.transform1 is not None:
            sample = self.transform1(image=sample)
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class TransformedSTL10(torchvision.datasets.STL10):
    def __init__(self, root, split='train', folds=None, transform=None, transform1=None, target_transform=None, download=False):
        super(TransformedSTL10, self).__init__(root, split, folds, transform, target_transform, download)
        self.transform1 = transform1

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        # img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # if self.transform is not None:
        #     img = self.transform(img)

        # img = np.array(img).transpose(1, 2, 0).astype(np.uint8)
        # if self.transform1 is not None:
        #     img = self.transform1(image=img)
        #     img = torch.from_numpy(img.transpose(2, 0, 1)).to(torch.float)
        
        # input img = 3 * 96 * 96
        img = np.transpose(img, (1, 2, 0)) # get 96 * 96 * 3
        if self.transform1 is not None: 
            img = self.transform1(image=img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        ## Filter abnormal lines such as NAN lines, empty lines, and change the type of each line to str
        df = df.dropna()
        # df['sentence1'], df['sentence2'] = df['sentence1'].astype(str), df['sentence2'].astype(str)
        if isinstance(df.iloc[0]['gold_label'], str):
            df = df[~(df['gold_label'].str.contains('-'))]
        df = df[(df['sentence1'].str.split().str.len() > 0) & (df['sentence2'].str.split().str.len() > 0)]
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_testing = is_testing
        self.label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    def __len__(self):
        return self.df.shape[0]

    # def __getitem__(self, idx):
    #     premise = self.df.iloc[idx]['sentence1']
    #     hypothesis = self.df.iloc[idx]['sentence2']
    #     result_item = self.tokenizer(premise, hypothesis, return_tensors='pt', max_length=self.max_length,
    #                                  padding='max_length', truncation=True)
    #     encoded_dict = {'input_ids': result_item['input_ids'].flatten(),
    #                     'attention_mask': result_item['attention_mask'].flatten(),
    #                     'token_type_ids': result_item['token_type_ids'].flatten(),
    #                    }
    #     if not self.is_testing:
    #         label = self.df.iloc[idx]['gold_label']
    #         encoded_dict['label'] = self.label_dict[label]
    #     return encoded_dict

    def __getitem__(self, idx):
        premise = self.df.iloc[idx]['sentence1']
        hypothesis = self.df.iloc[idx]['sentence2']
        premise_id = self.tokenizer.encode(premise)
        hypothesis_id = self.tokenizer.encode(hypothesis)

        input_ids = premise_id + hypothesis_id[1:]   ## <cls>+<premise_id>+<sep>+<hypothesis_id>+<sep>
        attn_mask = [1] * len(input_ids)             ## mask padded values
        token_type_ids = [0] * len(premise_id) + [1] * len(hypothesis_id[1:])  # sentence1->0 and sentence2->1

        # PAD
        pad_len = self.max_length - len(input_ids)
        input_ids += [0] * pad_len
        attn_mask += [0] * pad_len
        token_type_ids += [0] * pad_len

        input_ids, attn_mask, token_type_ids = map(torch.LongTensor, [input_ids, attn_mask, token_type_ids])

        encoded_dict = {
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'token_type_ids': token_type_ids,
        }
        if not self.is_testing:
            label = self.df.iloc[idx]['gold_label']
            encoded_dict['label'] = self.label_dict[label]
        return encoded_dict

class SICKBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(SICKBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}

class HANSBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(HANSBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {'entailment': 0, 'non-entailment': 1}

class QNLIBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(QNLIBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {'entailment': 0, 'not_entailment': 1}

class WNLIBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(WNLIBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {0: 0, 1: 1}

class ANLIBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(ANLIBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {'e': 0, 'n': 1, 'c': 2}

class SciTailBertDataset(BertDataset):
    def __init__(self, df, tokenizer, max_length, is_testing=False):
        super(SciTailBertDataset, self).__init__(df, tokenizer, max_length, is_testing=is_testing)
        self.label_dict = {'entails': 0, 'neutral': 1}


def set_seed_torch(seed=0):
    # Set seed based on args.seed and the update number so that we get
    # reproducible results when resuming from checkpoints
    assert isinstance(seed, int)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# forbidden to save as tar.gz
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.split(filename)[0] + '/checkpoint_best.pth')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def multi_acc(y_pred, y_test):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() * 100.0 / float(y_test.size(0))
  return acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
